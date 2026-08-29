"""Single-GPU training loop.

Upstream's `train_util.TrainLoop` is MPI-coupled at the seams -- `dist_util`,
rank-gated logging, a hand-rolled fp16 master-parameter copy from before
`torch.amp` existed. Rather than gut it in place (vendored code is read, not
rewritten) this is the same loop for one GPU: AdamW, linear warmup, EMA, grad
clipping, periodic checkpoints, and a held-out epsilon-loss monitor.

What it deliberately does NOT do is sample. Progress on a diffusion model is
judged by a fixed-seed sample grid plus held-out FID, and the ancestral sampler
is written by hand from the posterior q(x_{t-1} | x_t, x_0), not lifted from a
loop. So this saves EMA checkpoints and stops; sampling is a separate script
that loads one.
"""

import copy
import csv
import time
from pathlib import Path

import torch
from torch.optim import AdamW


class EMA:
    """Exponential moving average of the weights, kept as a shadow module.

    Sampling from the EMA weights rather than the raw iterate is not optional:
    it is the single most common cause of "my DDPM doesn't work". 0.9999 is
    Ho's value (DDPM appendix B) -- roughly a 10k-step averaging window, which
    means EMA samples are meaningless before ~10k steps and the shadow is
    initialised from the raw weights so early checkpoints are at least valid.
    """

    def __init__(self, model, rate):
        self.rate = rate
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.lerp_(p.detach(), 1.0 - self.rate)
        # No BatchNorm in this U-Net, so buffers are inert -- copied rather than
        # averaged so a future buffer-carrying block does not silently drift.
        for ema_b, b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(b)


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        val_batches=None,
        out_dir,
        batch_size,
        microbatch=-1,
        lr=1e-4,
        weight_decay=0.0,
        ema_rate=0.9999,
        grad_clip=1.0,
        warmup_steps=1000,
        lr_anneal_steps=0,
        total_steps=100_000,
        log_interval=100,
        eval_interval=2_000,
        save_interval=10_000,
        device="cuda",
        precision="fp32",
        resume=None,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.val_batches = val_batches or []
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.lr_anneal_steps = lr_anneal_steps
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.device = torch.device(device)
        self.precision = precision

        self.model.to(self.device)
        self.opt = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.ema = EMA(self.model, ema_rate)
        self.step = 0
        if resume:
            self._load(resume)

        self._log_path = self.out_dir / "log.csv"
        self._log_fields = ["step", "loss", "grad_norm", "lr", "sec_per_step",
                            "val_loss", "val_loss_ema"]
        if not self._log_path.exists():
            with open(self._log_path, "w", newline="") as f:
                csv.DictWriter(f, self._log_fields).writeheader()

    # -- schedules ---------------------------------------------------------

    def _lr_at(self, step):
        """Linear warmup, then optionally linear decay to zero.

        Warmup is not cosmetic here: at step 0 the network answers every noise
        level with garbage, and the largest gradients come from the high-t end
        where the target epsilon is pure noise.
        """
        lr = self.lr
        if self.warmup_steps > 0:
            lr *= min(1.0, (step + 1) / self.warmup_steps)
        if self.lr_anneal_steps > 0:
            lr *= max(0.0, 1.0 - step / self.lr_anneal_steps)
        return lr

    def _autocast(self):
        if self.precision == "bf16":
            return torch.autocast(device_type=self.device.type,
                                  dtype=torch.bfloat16)
        return torch.autocast(device_type=self.device.type, enabled=False)

    def _sample_t(self, n, generator=None):
        """Uniform timesteps, weight 1.

        Upstream's `resample.py` also offers loss-second-moment importance
        sampling. That exists to tame the variance of the *variational* terms,
        so it belongs with learn_sigma, not here: under L_simple every t already
        contributes O(1) because epsilon is unit-variance by construction.
        """
        return torch.randint(
            0, self.diffusion.num_timesteps, (n,),
            device=self.device, generator=generator,
        )

    # -- one optimizer step ------------------------------------------------

    def _train_step(self, batch, cond):
        lr = self._lr_at(self.step)
        for g in self.opt.param_groups:
            g["lr"] = lr

        self.opt.zero_grad(set_to_none=True)
        n = batch.shape[0]
        total = 0.0
        for i in range(0, n, self.microbatch):
            micro = batch[i : i + self.microbatch].to(self.device, non_blocking=True)
            micro_cond = {k: v[i : i + self.microbatch].to(self.device)
                          for k, v in cond.items()}
            t = self._sample_t(micro.shape[0])
            with self._autocast():
                losses = self.diffusion.training_losses(
                    self.model, micro, t, model_kwargs=micro_cond
                )
            # Scale by the microbatch's share so the gradient matches one pass
            # over the full batch regardless of how it was chunked.
            loss = losses["loss"].mean() * (micro.shape[0] / n)
            loss.backward()
            total += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip
        ).item()
        self.opt.step()
        self.ema.update(self.model)
        return total, grad_norm, lr

    # -- held-out monitor --------------------------------------------------

    @torch.no_grad()
    def _val_loss(self, module):
        """Mean epsilon-loss over the fixed val batches.

        Both the timesteps and the noise are drawn from a generator re-seeded
        identically at every evaluation, so consecutive points differ because
        the *weights* moved, not because a different (t, epsilon) was drawn.
        Without that the sampling noise swamps the train/val gap this is here
        to detect.

        The absolute value is uninformative -- L_simple's floor is irreducible
        conditional variance, so a flat curve is expected and is not a bug.
        Only train-vs-val divergence means anything.
        """
        if not self.val_batches:
            return None
        was_training = module.training
        module.eval()
        gen = torch.Generator(device=self.device).manual_seed(1234)
        total, count = 0.0, 0
        for batch, cond in self.val_batches:
            batch = batch.to(self.device)
            cond = {k: v.to(self.device) for k, v in cond.items()}
            t = self._sample_t(batch.shape[0], generator=gen)
            noise = torch.randn(batch.shape, device=self.device, generator=gen)
            with self._autocast():
                losses = self.diffusion.training_losses(
                    module, batch, t, model_kwargs=cond, noise=noise
                )
            total += losses["loss"].sum().item()
            count += batch.shape[0]
        if was_training:
            module.train()
        return total / count

    # -- checkpoints -------------------------------------------------------

    def _save(self):
        path = self.out_dir / f"ckpt_{self.step:07d}.pt"
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "ema": self.ema.module.state_dict(),
                "opt": self.opt.state_dict(),
            },
            path,
        )
        print(f"[save] {path}", flush=True)

    def _load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.ema.module.load_state_dict(ckpt["ema"])
        self.opt.load_state_dict(ckpt["opt"])
        self.step = ckpt["step"]
        print(f"[resume] {path} at step {self.step}", flush=True)

    def _log(self, row):
        with open(self._log_path, "a", newline="") as f:
            csv.DictWriter(f, self._log_fields).writerow(row)
        print(
            "  ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in row.items() if v is not None),
            flush=True,
        )

    # -- driver ------------------------------------------------------------

    def run(self):
        self.model.train()
        running, seen, t0 = 0.0, 0, time.time()
        saved_at = self.step
        while self.step < self.total_steps:
            batch, cond = next(self.data)
            loss, grad_norm, lr = self._train_step(batch, cond)
            self.step += 1
            running += loss
            seen += 1

            row = None
            if self.step % self.log_interval == 0:
                row = {
                    "step": self.step,
                    "loss": running / seen,
                    "grad_norm": grad_norm,
                    "lr": lr,
                    "sec_per_step": (time.time() - t0) / seen,
                    "val_loss": None,
                    "val_loss_ema": None,
                }
                running, seen, t0 = 0.0, 0, time.time()

            if self.eval_interval and self.step % self.eval_interval == 0:
                row = row or {"step": self.step, "loss": None, "grad_norm": None,
                              "lr": lr, "sec_per_step": None}
                row["val_loss"] = self._val_loss(self.model)
                row["val_loss_ema"] = self._val_loss(self.ema.module)
                t0 = time.time()

            if row is not None:
                self._log(row)

            if self.save_interval and self.step % self.save_interval == 0:
                self._save()
                saved_at = self.step

        if self.step != saved_at:
            self._save()
        print("[done]", flush=True)

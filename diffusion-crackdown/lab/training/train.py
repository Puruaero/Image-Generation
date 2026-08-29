"""Train a diffusion model on NIH ChestX-ray14. Single GPU.

Run from diffusion-crackdown/:

    python -m lab.training.train \
        --image_dirs /path/to/cxr14/images_001 ... /path/to/cxr14/images_012 \
        --train_list /path/to/cxr14/lists/train_list.txt \
        --val_list   /path/to/cxr14/lists/val_list.txt \
        --out_dir    runs/ddpm64_plain

Defaults are plain-DDPM mode -- linear schedule, fixed large variance, MSE --
which is Ho 2020 exactly. Improved DDPM is the same command plus
`--learn_sigma --noise_schedule cosine --rescale_learned_sigmas`, no code change.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lab.backends import get_backend
from lab.data.cxr14 import CXR14Dataset, load_data
from lab.training.loop import TrainLoop


def training_defaults():
    return dict(
        batch_size=64,
        microbatch=-1,          # -1: no gradient accumulation
        lr=1e-4,
        weight_decay=0.0,
        ema_rate=0.9999,
        grad_clip=1.0,
        warmup_steps=1000,
        lr_anneal_steps=0,      # 0: constant lr after warmup
        total_steps=200_000,
        log_interval=100,
        eval_interval=2_000,
        save_interval=10_000,
        num_workers=8,
        val_batches=16,         # batches of held-out data per eval
        precision="fp32",       # fp32 | bf16
        seed=0,
    )


def add_defaults(parser, defaults):
    for key, val in defaults.items():
        kind = type(val)
        if kind is bool:
            parser.add_argument(f"--{key}", type=str2bool, default=val,
                                nargs="?", const=True)
        else:
            parser.add_argument(f"--{key}", type=kind, default=val)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dirs", nargs="+", required=True)
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--val_list", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--backend", default="openai")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")

    # Two-pass parse: the backend decides which model flags exist.
    known, _ = parser.parse_known_args()
    backend = get_backend(known.backend)
    model_defaults = backend.defaults()
    add_defaults(parser, model_defaults)
    add_defaults(parser, training_defaults())
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    model_cfg = {k: getattr(args, k) for k in model_defaults}
    model, diffusion = backend.create_model_and_diffusion(**model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {known.backend} backend, {n_params/1e6:.1f}M params, "
          f"{args.image_size}^2 x {args.in_channels}ch, "
          f"T={diffusion.num_timesteps}", flush=True)

    data = load_data(
        image_dirs=args.image_dirs,
        batch_size=args.batch_size,
        resolution=args.image_size,
        split_list=args.train_list,
        num_workers=args.num_workers,
    )

    val_batches = None
    if args.val_list:
        # Materialised once and held in memory: the monitor must see the same
        # images every time, or the curve moves for reasons other than training.
        val_ds = CXR14Dataset(args.image_dirs, args.image_size,
                              split_list=args.val_list)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                drop_last=True)
        val_batches = []
        for i, batch in enumerate(val_loader):
            if i >= args.val_batches:
                break
            val_batches.append(batch)
        print(f"[data] val monitor: {len(val_batches)} batches "
              f"({len(val_batches) * args.batch_size} of {len(val_ds)} images)",
              flush=True)

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        val_batches=val_batches,
        out_dir=out_dir,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ema_rate=args.ema_rate,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        lr_anneal_steps=args.lr_anneal_steps,
        total_steps=args.total_steps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        device=args.device,
        precision=args.precision,
        resume=args.resume,
    ).run()


if __name__ == "__main__":
    main()

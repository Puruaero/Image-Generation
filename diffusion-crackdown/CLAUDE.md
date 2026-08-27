# CLAUDE.md — diffusion-crackdown

Diffusion models, implemented from the papers, on medical images. Parent `CLAUDE.md` in
`Image-Generation/` carries working style, git auth, and vendoring conventions — read it too.

```
diffusion-crackdown/
└── openai_diffusion/         # openai/improved-diffusion, git subtree, MIT LICENSE retained
    ├── improved_diffusion/   # gaussian_diffusion.py, unet.py, respace.py, train_util.py
    ├── scripts/
    └── datasets/
```

**Status: milestone 1 — DDPM producing recognizable samples at 64² on NIH ChestX-ray14.**
Nothing trains yet. First code task is the dataloader.

---

## The design rule — this is the whole architecture

One noise-level-conditioned network. Four swappable things around it:

- **corruption** — VP (DDPM) / VE (SMLD)
- **sampler** — ancestral / DDIM / annealed Langevin
- **space** — pixel / latent
- **conditioning** — none / class / text

Every paper after DDPM should be a **config change, not a rewrite.** If a paper forces a
rewrite, the abstraction is wrong — fix the abstraction, not the paper. Judge every proposed
code change against this before writing it.

`improved-diffusion` was chosen on exactly this criterion, not on day-one ergonomics:
`gaussian_diffusion.py` already factors the parameterization axis into enums —
`ModelMeanType` (eps / x0 / x_prev), `ModelVarType` (fixed_small / fixed_large / learned /
learned_range), `LossType` (mse / rescaled_mse / kl). Config, not class hierarchy.
`respace.py` gives DDIM and arbitrary timestep subsequences over unchanged weights.
Cosine schedule, learned Σ, hybrid loss, EMA, warmup, grad-clip, fp16 all present.

What it does **not** have is VE / annealed Langevin. That gap is deliberate — SMLD is the
part he writes himself, against a working VP baseline.

Rejected and why: `lucidrains` (VP-only monolithic `GaussianDiffusion` — keep as a U-Net
reading reference only); `score_sde_pytorch` (right abstraction, JAX-transliterated
ergonomics — read it at core step 3, don't bring up CXR on it); MONAI (wraps the scheduler,
which is the exact layer being learned).

---

## Build order

**Core — sequential, pixel space:**

| # | Paper | What changes | Cost |
|---|-------|--------------|------|
| 1 | DDPM (Ho 2020) | everything | the real build ← **current** |
| 2 | DDIM (Song 2020) | sampler only | `--timestep_respacing ddim50`, weights unchanged |
| 3 | Improved DDPM (Nichol & Dhariwal) | cosine schedule, learned Σ | flags only |
| 4 | EDM (Karras 2022) | reframes 1–3 | rewrites interfaces, not the network |

EDM is not another model — it's the paper that says which knobs in 1–3 were arbitrary all
along. Read it **after** those knobs have annoyed him personally.

**Side tracks, one step each, slot in anywhere:**

- **A. VAE.** ELBO baseline on the same dataset, plus encoder intuition for latent diffusion
  later. Half-done in theory already — DDPM's bound *is* the VAE ELBO extended to a T-step
  hierarchy with a fixed encoder.
- **B. SMLD → Score SDE.** SMLD enters the **core codebase** as a corruption + sampler swap
  (VE + annealed Langevin), not a second repo. Then Score SDE, where VE/VP drop to config
  flags. **Do B after DDPM stands** — annealed Langevin is finicky and needs to be debugged
  against something known to work.

**Axes — after the core, one variable at a time:**

1. **Conditioning** — unconditional → class-conditional → CFG. Still pixel space, still the
   same U-Net. CXR hands you free labels (findings) now and free text (MIMIC reports) later.
   `openai/guided-diffusion` is the same file tree plus conditioning, so this is a readable
   diff rather than a redesign.
2. **Space** — Latent Diffusion. **After** CFG, deliberately: LDM turns on space *and*
   conditioning simultaneously, and splitting them means never debugging two new things at
   once. **Freeze a pretrained autoencoder — do not train one.** SD's "VAE" is a
   KL-weight-1e-6 near-deterministic autoencoder trained with LPIPS + a patch GAN
   discriminator; reproducing it is a multi-week GAN slog that teaches nothing about
   diffusion. Medical caveat: natural-image autoencoders reconstruct CXR imperfectly — check
   reconstruction quality before trusting anything downstream. MONAI ships medical
   AutoencoderKL weights.
3. **Architecture** — DiT. Lands cleanly last because DiT already runs in SD's latent space,
   so by then only the architecture moves. (Note: "EDM via transformers" is not a paper —
   EDM2 is a U-Net training-dynamics paper. DiT is the transformer entry point.)

Scope bar: **stay inside diffusion.** Flow matching / rectified flow is parked — same
continuous-time family, a later room in the same house. An afternoon whenever he wants it.

---

## Data — NIH ChestX-ray14

~112k frontal CXR, 1024² PNG, 30,805 patients, 14 finding labels, fully open. Downloaded.
MIMIC-CXR (~377k + reports) is the later/serious corpus if PhysioNet credentialing goes
through; nothing else in the plan changes if it does.

Three non-negotiables:

- **Patient-level split.** Multiple studies per patient — a random image split puts the same
  chest in train and test and quietly inflates every held-out number. Use the shipped
  `train_val_list.txt` / `test_list.txt`, which are already patient-separated. Do not roll
  a random split.
- **No horizontal flip.** It mirrors the cardiac silhouette and the L/R markers, so the model
  learns to generate anatomically impossible chests. It is the default in essentially every
  reference repo, `improved-diffusion` included — **grep the dataset class and remove it.**
- **Grayscale, 64² first.** `ImageDataset` assumes RGB + center-crop and needs rewiring. A
  failure at 256² is unattributable.

Linear map to [−1,1], no mean subtraction, no whitening. There is no first-principles
argument for [−1,1] specifically — what *is* forced is O(1) scale, because the forward
process is not scale-invariant: SNR(t) = c²·ᾱ_t/(1−ᾱ_t) for data scaled by c. **Data scale
and the β_t schedule are one knob, not two** — rescaling the data and shifting the noise
schedule are the same intervention. This resurfaces at higher resolution, where neighboring-
pixel redundancy raises effective SNR and the same schedule becomes too weak.

**Why a split at all, for a generative model** — three uses, none of them "test accuracy":
held-out ε-loss as an overfitting monitor (train-vs-val divergence is the signal); FID/KID
against images the model never saw (FID against the training set is optimistically biased
and not comparable to published numbers); memorization check (sample N, nearest-neighbour
back into the *training* set — near-copies mean a retrieval system, not a generative model;
low-priority at 112k, real if he ever subsets down).

---

## Bring-up recipe — milestone 1

Start in **plain-DDPM mode**: `learn_sigma=False`, `noise_schedule=linear`, MSE loss,
`FIXED_LARGE`. That is Ho 2020 exactly. Then flip to Improved DDPM by changing flags only —
a live test of the config-change-not-rewrite rule on the very first paper transition.

ε-prediction U-Net, T=1000. AdamW 1e-4–2e-4, warmup, grad-clip 1.0.
**EMA 0.999, and sample from the EMA weights** — the single most common cause of
"my DDPM doesn't work."

**Judge progress by a fixed-seed sample grid every N epochs plus held-out FID.** Not by the
loss curve: L_simple plateaus early and stays uninformative, because its floor is
*irreducible conditional variance* — the exact ε is unrecoverable (many (x_0, ε) pairs give
the same x_t), so the L2 minimizer is E[ε|x_t], not ε. **A flat loss is not a bug and does
not need diagnosing.**

**The ancestral sampler is his to write**, from the posterior q(x_{t−1}|x_t, x_0) he already
derived — not copied from a loop. Variance-term bugs there are the classic silent failure:
oversmoothed or noisy samples, and nothing else reports it. Do not write it for him.

---

## Repo friction, budgeted (it's from 2021)

`mpi4py` / `dist_util` — strippable for single-GPU. `blobfile`. Old torch idioms around
`torch.load` and AMP. `ImageDataset` assumes RGB + center-crop.

**Reading order, agreed — do not reorder:**

1. `improved_diffusion/gaussian_diffusion.py` — `q_sample`, `q_posterior_mean_variance`,
   `p_mean_variance`, `training_losses`
2. `improved_diffusion/unet.py` — timestep embedding into ResBlocks
3. `improved_diffusion/respace.py` — only after DDPM works
4. `improved_diffusion/train_util.py` — EMA

---

## Theory already settled — do not re-explain unprompted

~80–85% on Sohl-Dickstein 2015, DDPM, and SMLD, at derivation level. He can rederive these.

- DDPM's ELBO collapses to weighted noise regression (each L_{t−1} is a KL between Gaussians
  with the same known variance → pure mean matching); L_simple drops the weight.
- **Denoising = score estimation.** E[ε|x_t] = −√(1−ᾱ_t)·∇ log q(x_t). The optimal denoiser
  is the score of the noised marginal in disguise.
- **ε_θ := −σ_t·s_θ is a reparameterization, not an optimum-only equivalence** — it holds
  inside the objective as a change of variables. A network trained on L_simple **is** a score
  network up to that factor, convertible either direction at any point in training. Sampling
  needs only one of them: ancestral → ε_θ; Langevin / reverse-SDE → s_θ = −ε_θ/σ_t. **Same
  network.** The code should make this load-bearing rather than incidental.
- **σ² is the natural equalizer.** ‖score‖ ~ 1/σ, so σ² makes every noise level contribute
  O(1). Equivalently: ε is unit-variance by construction at every level, so predicting ε
  rather than the score *is* the σ² weighting, absorbed into the parameterization. Song chose
  it by design; Ho arrived at the same place by dropping an ELBO coefficient.
- **The only structural DDPM/SMLD difference is VP vs VE.** DDPM shrinks the signal and
  preserves variance; SMLD leaves the signal unscaled and lets variance explode. Song 2021
  formalizes both as SDEs — that is the next paper on the theory track, running in parallel
  with this code.
- No unit-variance-data assumption is baked into the DDPM derivations. Everything is
  conditioned on x_0, hence distribution-free in x_0; Var(x_0)=I matters only for the
  *marginal* recursion, i.e. for the word "variance-preserving," not for any derivation.

Open, not done: **the angles** — score field vs ε-field vs E[x_0|x_t] (Tweedie) as one
vector field in three scalings, drawn once at one noise level.

---

## Reference repos

- `lucidrains/denoising-diffusion-pytorch` — clean DDPM, U-Net reading reference
- `openai/guided-diffusion` — same file tree + conditioning/guidance; axis 2
- `yang-song/score_sde_pytorch` — VE/VP as config switches; north star for the abstraction
- `ermongroup/ncsnv2` — original SMLD tricks
- `hojonathanho/diffusion` — ground truth for constants
- `AntixK/PyTorch-VAE` — VAE baseline
- `Project-MONAI/GenerativeModels` (merged into MONAI core 1.3+) — medical AutoencoderKL
  weights, BraTS + CXR tutorials
- `mueller-franzes/medfusion`, `Warvito/generative_monai` — medical diffusion prior art

**HF `diffusers` is a correctness oracle only — never the learning vehicle.**

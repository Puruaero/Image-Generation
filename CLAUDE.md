# CLAUDE.md — Image-Generation

Personal research repo. One goal: understand generative models by **implementing them from
the papers**, not by calling a library. Each major technique family gets its own top-level
folder with its own `CLAUDE.md` carrying the specifics.

This file is the part that stays true across all of them. Read the subfolder's `CLAUDE.md`
for whatever is actually being worked on.

```
Image-Generation/
├── CLAUDE.md                 # this file — conventions, working style, git
└── diffusion-crackdown/      # diffusion models (active) — see its own CLAUDE.md
    ├── openai_diffusion/     # openai/improved-diffusion, vendored as a git subtree
    └── lab/                  # his own layer: data, training, samplers, backend adapters
```

**One folder per paradigm.** `diffusion-crackdown/` is denoising diffusion only; score
matching (SMLD, Score SDE) goes in `score-matching-crackdown/`, VAEs in their own, and so
on. The split is by paradigm rather than by convenience — a folder whose corruption family
or objective family differs is a different folder, even when it shares the dataset and the
theory track. Future folders (score matching, VAEs standalone, flow matching, point-cloud
generation, GANs if ever) follow the same shape: one folder, one `CLAUDE.md`, vendored upstream code kept in a clearly named
subfolder with its original LICENSE intact, and **his own code in a sibling `lab/`** rather
than mixed into the vendored tree. The vendored core is a dependency of `lab/`, never the
other way round — that is what keeps it swappable.

---

## Working style — read this first, it applies everywhere

- **He derives the math himself and enjoys it.** When he says "I'll work through it," give
  the setup and the destination, not the walkthrough. Do not hand him a derivation he did
  not ask for. This is the single most important line in this file.
- **Derive forward**, in the order the objects actually become available. Never state a
  result and justify it backwards. He reads line by line and back-calculated flow breaks
  the reading.
- **LaTeX does not render in his client.** Use Unicode math inline in prose (ᾱ_t, σ_t, √,
  ‖·‖², E, ∇). Do not fall back to fenced code blocks for math — the boxing makes it worse,
  not better.
- Directness. No flattery, no over-hedging, no "great question." He pushes back on
  imprecision and would rather be told something is wrong than be managed.
- **Vendored upstream code is read, not rewritten.** Architecture and training loops come
  from the reference repo. The pieces that constitute the actual learning — samplers,
  objectives, anything he has derived on paper — are **his to write**. Do not write those
  for him and do not paste an implementation in "for reference."
- If asked to explain something, assume he has already read the paper.

---

## Git — two GitHub accounts on one machine

This machine is used with two GitHub accounts: a work account and a personal one. This repo
belongs to the personal account (`Puruaero`). Standard multi-account setup — the two are kept
on separate credentials so git always picks the right identity per repo instead of guessing.

- **Work account:** HTTPS with the system credential helper. Configured already; nothing here
  changes it, and no global git or ssh setting should be modified for this repo's sake.
- **This repo:** a dedicated SSH host alias, which is the conventional way to pin a second
  account. `Host github-personal` in `~/.ssh/config` with `HostName github.com`,
  `IdentityFile ~/.ssh/id_personal`, and **`IdentitiesOnly yes`** — without that last line ssh
  offers the other key first and the push is rejected. Remote is
  `git@github-personal:Puruaero/Image-Generation.git`.
  `user.email` / `user.name` are set **repo-local**, so commits here are attributed correctly
  without changing anything globally.

**If a push fails with `403 Permission to Puruaero/Image-Generation.git denied to
purunfer22`, that is simply the wrong account being offered** — the alias or `IdentitiesOnly`
is not in effect. Fix it repo-locally; changing global config would fix this repo and break
the other one.

Work repos on this machine are unaffected by any of the above and should stay that way.

## Vendoring convention

Upstream repos come in as **git subtrees**, not forks or file copies (GitHub forks inherit
parent visibility, and this repo is private):

```
git remote add upstream <url>
git fetch upstream
git subtree add  --prefix=<folder>/<name> upstream main
git subtree pull --prefix=<folder>/<name> upstream main   # later syncs
```

Upstream history is preserved, the original LICENSE stays inside the folder, and the
README carries a provenance line. Local modifications to vendored code are fine and
expected — but they should be small, commented, and worth the merge cost on the next pull.

---

## Sync note

The full project plan and the running theory notes live in the Cowork project
"Diffusion Models Crackdown," which Claude Code sessions cannot attach to. These CLAUDE.md
files are the condensed mirror. **They drift.** When a decision changes in a code session,
say so at the end of it so the project docs get updated there — don't let the two versions
diverge silently.

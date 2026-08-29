"""Backends — the swappable diffusion core.

`lab` owns the data, the training loop and (later) the samplers. What it does
*not* own is the network and the forward/reverse process: those come from a
vendored reference implementation, read rather than rewritten.

A backend is any module exposing two functions:

    defaults()                        -> dict of config keys with default values
    create_model_and_diffusion(**cfg) -> (nn.Module, diffusion)

where `diffusion` provides `.training_losses(model, x_0, t, model_kwargs)` and
`.num_timesteps`. That is the whole contract. Swapping `openai_diffusion` for a
different core (score_sde, a from-scratch VE process) means adding one module
here, not touching training/ or data/.
"""

import importlib

BACKENDS = {
    "openai": "lab.backends.openai",  # openai/improved-diffusion, vendored
}


def get_backend(name):
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; have {sorted(BACKENDS)}")
    return importlib.import_module(BACKENDS[name])

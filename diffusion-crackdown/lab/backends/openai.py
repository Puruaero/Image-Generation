"""Adapter for the vendored openai/improved-diffusion core.

Thin on purpose. It re-implements `script_util.create_model_and_diffusion`
rather than calling it for exactly one reason: upstream hardcodes
`in_channels=3` and `out_channels=3 (or 6)` inside `create_model`, so there is
no flag that gets you a single-channel U-Net. Everything else here is upstream's
own construction, unchanged -- channel_mult per resolution, attention
downsample rates, and `create_gaussian_diffusion` called directly.

Keep it that way. If a future need can be expressed as a config key upstream
already has, pass the key; do not fork more of script_util into this file.
"""

import sys
from pathlib import Path

# The vendored repo is a sibling directory, not an installed package. Putting it
# on sys.path here keeps `pip install -e` out of the loop and confines the hack
# to the one module that depends on this backend existing.
_VENDOR = Path(__file__).resolve().parents[2] / "openai_diffusion"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from improved_diffusion.script_util import create_gaussian_diffusion  # noqa: E402
from improved_diffusion.unet import UNetModel  # noqa: E402

# Downsampling schedule per resolution, copied from upstream create_model.
CHANNEL_MULT = {
    32: (1, 2, 2, 2),
    64: (1, 2, 3, 4),
    256: (1, 1, 2, 2, 4, 4),
}


def defaults():
    """Plain-DDPM mode: Ho 2020 exactly.

    learn_sigma=False + sigma_small=False -> FIXED_LARGE, and with
    rescale_learned_sigmas=False the loss type is plain MSE, i.e. L_simple.
    Improved DDPM is reached from here by flags alone:
    --learn_sigma --noise_schedule cosine --rescale_learned_sigmas.
    """
    return dict(
        image_size=64,
        in_channels=1,          # grayscale; upstream has no flag for this
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )


def create_model_and_diffusion(
    *,
    image_size,
    in_channels,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    learn_sigma,
    sigma_small,
    class_cond,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
):
    if image_size not in CHANNEL_MULT:
        raise ValueError(f"unsupported image size: {image_size}")
    if class_cond:
        # Conditioning is axis 1, after the unconditional model works. Wiring it
        # means deciding what num_classes means for 14 non-exclusive findings,
        # which is a design question, not a plumbing one.
        raise NotImplementedError("class conditioning not wired up yet")

    attention_ds = [image_size // int(r) for r in attention_resolutions.split(",")]

    model = UNetModel(
        in_channels=in_channels,
        model_channels=num_channels,
        # Learned Sigma doubles the output: (eps, v) per channel.
        out_channels=in_channels * (2 if learn_sigma else 1),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=CHANNEL_MULT[image_size],
        num_classes=None,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion

"""NIH ChestX-ray14 dataloader.

Replaces improved_diffusion.image_datasets.ImageDataset, which assumes RGB and
pulls in MPI. Three things here are deliberate and not to be "fixed":

  * grayscale, single channel  -- CXR are single-channel; RGB triples the input
    for no information.
  * no horizontal flip         -- mirroring flips the cardiac silhouette and the
    L/R markers, so the model learns anatomically impossible chests.
  * split by the shipped lists -- train_val_list.txt / test_list.txt are already
    patient-separated. A random image split puts the same patient on both sides.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg"}

# Pillow moved these onto an enum in 9.1; the top-level aliases warn on some
# versions. Works either side of that.
_RESAMPLE = getattr(Image, "Resampling", Image)


def _index_images(image_dirs):
    """All image files across the given directories, sorted by filename.

    NIH ships images across images_001/images/ ... images_012/images/, so this
    takes a list of directories rather than a single root. Recurses, so passing
    the twelve images_XXX/ folders works as well as their images/ subfolders.
    """
    paths = []
    for d in image_dirs:
        d = Path(d)
        if not d.is_dir():
            raise FileNotFoundError(f"not a directory: {d}")
        paths.extend(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise FileNotFoundError(f"no images found under: {list(image_dirs)}")

    # Sort by basename, not full path: the split lists are basenames, and the
    # ordering should not depend on which images_XXX/ folder a file landed in.
    paths.sort(key=lambda p: p.name)

    names = [p.name for p in paths]
    if len(set(names)) != len(names):
        raise ValueError("duplicate image basenames across the given directories")
    return paths


def _read_split(split_list):
    """Basenames listed in train_val_list.txt / test_list.txt."""
    with open(split_list) as f:
        return {line.strip() for line in f if line.strip()}


class CXR14Dataset(Dataset):
    def __init__(self, image_dirs, resolution, split_list=None):
        super().__init__()
        self.resolution = resolution
        paths = _index_images(image_dirs)

        if split_list is not None:
            keep = _read_split(split_list)
            paths = [p for p in paths if p.name in keep]
            if not paths:
                raise ValueError(
                    f"no images under {list(image_dirs)} appear in {split_list}"
                )
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as img:
            img.load()
            # A handful of NIH files are RGBA rather than 8-bit gray.
            img = img.convert("L")

            # Downsample in BOX steps at powers of two before the final resize;
            # a single 1024 -> 64 bicubic aliases badly. (Same trick as the
            # vendored ImageDataset.)
            while min(img.size) >= 2 * self.resolution:
                img = img.resize(tuple(x // 2 for x in img.size), _RESAMPLE.BOX)

            scale = self.resolution / min(img.size)
            img = img.resize(
                tuple(round(x * scale) for x in img.size), _RESAMPLE.BICUBIC
            )
            arr = np.array(img)

        # Center crop. A no-op for NIH's square 1024s, but the resize above only
        # guarantees the short side, so a non-square file would be left long.
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]

        # Linear map to [-1, 1]. No mean subtraction, no whitening: the forward
        # process is not scale-invariant, so data scale and the beta schedule are
        # one knob. Changing this means revisiting the schedule.
        arr = arr.astype(np.float32) / 127.5 - 1.0

        # (1, H, W); empty dict keeps the (batch, cond) contract train_util wants.
        return arr[None], {}


def load_data(*, image_dirs, batch_size, resolution, split_list=None,
              deterministic=False, num_workers=4):
    """Infinite generator over (images, cond) batches.

    Same contract as improved_diffusion.image_datasets.load_data, minus MPI.
    """
    dataset = CXR14Dataset(image_dirs, resolution, split_list=split_list)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    while True:
        yield from loader


def _main():
    """Sanity check against the real data: counts, shapes, pixel range."""
    p = argparse.ArgumentParser()
    p.add_argument("image_dirs", nargs="+")
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--split_list", default=None)
    p.add_argument("--n", type=int, default=64, help="images to sample for stats")
    args = p.parse_args()

    ds = CXR14Dataset(args.image_dirs, args.resolution, split_list=args.split_list)
    print(f"{len(ds)} images @ {args.resolution}^2")

    idx = np.random.default_rng(0).choice(len(ds), size=min(args.n, len(ds)),
                                          replace=False)
    batch = np.stack([ds[int(i)][0] for i in idx])
    print(f"shape {batch.shape}  dtype {batch.dtype}")
    print(f"range [{batch.min():.3f}, {batch.max():.3f}]  "
          f"mean {batch.mean():.3f}  std {batch.std():.3f}")

    patients = {p.name.split("_")[0] for p in ds.paths}
    print(f"{len(patients)} distinct patients")


if __name__ == "__main__":
    _main()

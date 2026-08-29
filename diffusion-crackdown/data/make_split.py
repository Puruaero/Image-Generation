"""Repartition NIH ChestX-ray14 into train / val / test lists by image count.

NIH ships train_val_list.txt and test_list.txt and no validation list. This pools
whatever lists you hand it and re-cuts them to target sizes (default 100k train,
5k val, remainder test).

That deliberately discards NIH's official train/test boundary, so numbers here
are not comparable to published CXR14 benchmarks -- irrelevant for generative
work, where nobody is scoring you against that leaderboard.

What does NOT bend: the cut is over *patients*, never images. A patient
contributes several studies of the same chest, so an image-level cut puts
near-duplicates on both sides and every held-out number silently inflates.

Output is NIH's format (one basename per line), so the lists drop straight into
cxr14.py's --split_list.
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path


def patient_id(name):
    """NIH names are <patientID>_<studyID>.png."""
    return name.split("_")[0]


def read_lists(paths):
    names = []
    for p in paths:
        with open(p) as f:
            names.extend(line.strip() for line in f if line.strip())
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate entries across {[str(p) for p in paths]}")
    return names


def split_by_patient(names, n_train, n_val, seed):
    """Cut into (train, val, test) at patient boundaries, targeting image counts.

    Patients are indivisible, so a bucket closes on the patient that first takes
    it to its target -- counts land near the targets, not exactly on them. The
    actual sizes are reported rather than forced.
    """
    by_patient = defaultdict(list)
    for n in names:
        by_patient[patient_id(n)].append(n)

    patients = sorted(by_patient)  # sort first: dict order must not affect the seed
    random.Random(seed).shuffle(patients)

    if n_train + n_val >= len(names):
        raise ValueError(
            f"n_train + n_val = {n_train + n_val} leaves no test images "
            f"out of {len(names)}"
        )

    buckets, targets = [[], [], []], [n_train, n_val, None]
    b, count = 0, 0
    for p in patients:
        # Close the current bucket once it has reached its target, then move on.
        # targets[-1] is None: test absorbs whatever is left.
        while targets[b] is not None and count >= targets[b]:
            b, count = b + 1, 0
        buckets[b].append(p)
        count += len(by_patient[p])

    train, val, test = (
        sorted(n for p in bucket for n in by_patient[p]) for bucket in buckets
    )
    if not val or not test:
        raise ValueError("a split came out empty -- check n_train / n_val")
    return train, val, test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("lists", nargs="+",
                   help="NIH lists to pool, e.g. train_val_list.txt test_list.txt")
    p.add_argument("--out_dir", default=".")
    p.add_argument("--n_train", type=int, default=100_000)
    p.add_argument("--n_val", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    names = read_lists(args.lists)
    train, val, test = split_by_patient(names, args.n_train, args.n_val, args.seed)

    # The whole point of the file -- assert it rather than trust it.
    pats = [{patient_id(n) for n in s} for s in (train, val, test)]
    for i, j, label in ((0, 1, "train/val"), (0, 2, "train/test"), (1, 2, "val/test")):
        assert not (pats[i] & pats[j]), f"patient leakage between {label}"
    assert len(train) + len(val) + len(test) == len(names), "images lost or duplicated"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total = len(names)
    print(f"pooled {total} images, {len(set().union(*pats))} patients "
          f"(seed {args.seed})")
    for name, rows, pat in (("train_list.txt", train, pats[0]),
                            ("val_list.txt", val, pats[1]),
                            ("test_list.txt", test, pats[2])):
        (out / name).write_text("\n".join(rows) + "\n")
        print(f"  {name:<15} {len(rows):>7} images ({len(rows)/total:5.1%})  "
              f"{len(pat):>6} patients")
    print("patient overlap between every pair: 0")


if __name__ == "__main__":
    main()

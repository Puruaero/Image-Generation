"""Repartition NIH ChestX-ray14 into train / val (/ test) lists by patient count.

NIH ships train_val_list.txt and test_list.txt and no validation list. This pools
whatever lists you hand it and re-cuts them: --val_patients held out, everything
else to train. Pass --test_patients to get a third list; omit it and only two are
written.

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


def split_by_patient(names, n_val, n_test, seed):
    """Cut into {train, val[, test]} at patient boundaries, by patient count.

    Sizes are exact in patients; the resulting image counts follow from however
    many studies those patients happen to have.
    """
    by_patient = defaultdict(list)
    for n in names:
        by_patient[patient_id(n)].append(n)

    patients = sorted(by_patient)  # sort first: dict order must not affect the seed
    random.Random(seed).shuffle(patients)

    if n_val + n_test >= len(patients):
        raise ValueError(
            f"val+test = {n_val + n_test} patients leaves none for train "
            f"out of {len(patients)}"
        )
    if n_val <= 0:
        raise ValueError("--val_patients must be positive")

    held = {"val": patients[:n_val]}
    if n_test:
        held["test"] = patients[n_val : n_val + n_test]
    held["train"] = patients[n_val + n_test :]

    return {
        split: sorted(n for p in pats for n in by_patient[p])
        for split, pats in held.items()
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("lists", nargs="+",
                   help="NIH lists to pool, e.g. train_val_list.txt test_list.txt")
    p.add_argument("--out_dir", default=".")
    p.add_argument("--val_patients", type=int, default=750)
    p.add_argument("--test_patients", type=int, default=0,
                   help="0 (default) writes no test list -- train/val only")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    names = read_lists(args.lists)
    splits = split_by_patient(names, args.val_patients, args.test_patients, args.seed)

    # The whole point of the file -- assert it rather than trust it.
    pats = {s: {patient_id(n) for n in rows} for s, rows in splits.items()}
    for a in pats:
        for b in pats:
            if a < b:
                assert not (pats[a] & pats[b]), f"patient leakage between {a}/{b}"
    assert sum(len(r) for r in splits.values()) == len(names), \
        "images lost or duplicated"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total = len(names)
    print(f"pooled {total} images, {len(set().union(*pats.values()))} patients "
          f"(seed {args.seed})")
    for split in ("train", "val", "test"):
        if split not in splits:
            continue
        rows = splits[split]
        (out / f"{split}_list.txt").write_text("\n".join(rows) + "\n")
        print(f"  {split + '_list.txt':<15} {len(rows):>7} images ({len(rows)/total:5.1%})"
              f"  {len(pats[split]):>6} patients")
    # A leftover list from an earlier run with different settings is the exact
    # silent leak this file exists to prevent -- its patients are now in train.
    stale = [f for f in ("train_list.txt", "val_list.txt", "test_list.txt")
             if f.replace("_list.txt", "") not in splits and (out / f).exists()]
    if stale:
        print(f"\nWARNING: {', '.join(stale)} in {out} is left over from an "
              f"earlier run and was NOT rewritten.\n"
              f"         Its patients are in train_list.txt now. Delete it.")

    print("patient overlap between every pair: 0")


if __name__ == "__main__":
    main()

"""Add the in-distribution LIBERO-10 rollouts of a seed pair (calibration and test) to its LIBERO-Plus
*scored* run, so that one directory holds the full failure-detection dataset of the pair.

Every vanilla rollout gets `--episode-offset` (default 1000) added to its episode number, which keeps
it apart from the LIBERO-Plus test rollouts (0-179 per task: family_index * 30 + j) and from an earlier
merge; it is tagged rollout_subtype "id" and benchmark "libero_vanilla". Both recordings use the same
number of uncertainty sequences K; `--k` keeps the first K action samples of every step (a no-op when
they already agree).

    python scripts/failure_detection/merge_libero_into_plus.py \
        outputs/fiper_rollout_scoring/smolvla_libero_s01/libero_10 \
        outputs/fiper_rollout_scoring/smolvla_libero_plus_s01/libero_10 --k 16
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

SAMPLE_AXIS = {"action_pred": 0, "velocities": 1, "ensemble_velocities": 2}


def slice_step(step: dict, k: int) -> dict:
    out = dict(step)
    for key, ax in SAMPLE_AXIS.items():
        if key in out:
            a = np.asarray(out[key])
            if a.shape[ax] < k:
                raise ValueError(f"{key} has only {a.shape[ax]} samples, need {k}")
            out[key] = np.take(a, range(k), axis=ax)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vanilla_root")
    ap.add_argument("plus_root")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--episode-offset", type=int, default=1000,
                    help="added to the vanilla episode index so the numbering cannot collide with the "
                         "LIBERO-plus rollouts (0-179) or with an earlier splice (default 1000)")
    ap.add_argument("--splits", default="calibration,test")
    args = ap.parse_args()
    van, plus = Path(args.vanilla_root), Path(args.plus_root)
    n_cal = n_test = 0
    for t in (int(x) for x in args.tasks.split(",")):
        for split in (x for x in args.splits.split(",") if x):
            src = van / f"task{t:02d}" / "rollouts" / split
            dst = plus / f"task{t:02d}" / "rollouts" / split
            dst.mkdir(parents=True, exist_ok=True)
            for f in sorted(src.glob("*.pkl")):
                d = pickle.load(open(f, "rb"))
                md = dict(d["metadata"])
                md["action_batch_size"] = args.k
                md.setdefault("rollout_subtype", "id")
                md["benchmark"] = "libero_vanilla"
                # keep vanilla episode numbering out of the LIBERO-plus range (>= 1000)
                ep = int(md.get("episode", 0)) + args.episode_offset
                md["episode"] = ep
                out = {"metadata": md, "rollout": [slice_step(s, args.k) for s in d["rollout"]]}
                if "config" in d:
                    out["config"] = d["config"]
                flag = "s" if md.get("successful") else "f"
                with open(dst / f"episode_{flag}_{ep:04d}_task{t:02d}.pkl", "wb") as fh:
                    pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
                if split == "calibration":
                    n_cal += 1
                else:
                    n_test += 1
    print(f"spliced {n_cal} calibration + {n_test} ID test rollouts at K={args.k} "
          f"(episode offset {args.episode_offset}) into {plus}")


if __name__ == "__main__":
    main()

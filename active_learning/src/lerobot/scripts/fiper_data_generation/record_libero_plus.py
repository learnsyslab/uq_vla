"""Record FIPER test rollouts on LIBERO-plus variants of LIBERO-10.

One process loads the policy once and loops over (base task, perturbation family) cells; every cell gets
`--n-per-cell` rollouts spread over the family's variants (one rollout per variant, cycling with a new
init state when a family has fewer variants). Output mirrors the vanilla recorder exactly --
<output_dir>/libero_10/taskNN/rollouts/test/episode_{s|f}_XXXX_taskNN.pkl with NN = the *vanilla*
LIBERO-10 task id -- so score_fiper_rollout.py and the failure-detection pipeline run unchanged. Rollouts are tagged
rollout_subtype "ood" plus `perturbation` / `variant` / `libero_plus_task_id` in the metadata.

Needs the LIBERO-plus fork on PYTHONPATH (it replaces `libero`; see scripts/setup_libero_plus.sh). Two
fork quirks are handled here: init states are resolved through the suite's own method (it strips the
variant suffix), and the instruction is read from the bddl instead of the fork's file-name heuristic,
which leaks the suffix into the prompt ("... on it table 1") and breaks the policy (0/30 -> 19/30).

    PYTHONPATH=src:<fork> python src/lerobot/scripts/fiper_data_generation/record_libero_plus.py \\
        --policy_path <member_00> --output_dir outputs/fiper_rollout_recording/smolvla_libero_plus_s01 \\
        --seed 20118 --shard 0 --num_shards 4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SUITE = "libero_10"
FAMILIES = ("camera", "robot_init", "noise", "background", "light", "layout")
ODE_EVAL_TIMES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]  # = the LIBERO recording configs


def classify(name: str) -> str:
    if "_noise_" in name:
        return "noise"
    if "_language_" in name:
        return "language"
    if re.search(r"_(?:table|tb)_\d+", name):
        return "background"
    if "_light_" in name:
        return "light"
    if re.search(r"_add_\d+|_level\d+_sample\d+", name):
        return "layout"
    m = re.search(r"_view_(\-?\d+)_(\-?\d+)_(\d+)_(\-?\d+)_(\-?\d+)_initstate_(\d+)", name)
    if m:
        h, v, scale, rot, vert, init = (int(x) for x in m.groups())
        if not (h == 0 and v == 0 and scale == 100 and rot == 0 and vert == 0):
            return "camera"
        return "robot_init" if init != 0 else "base"
    return "base"


def base_stem(name: str) -> str:
    for pat in (r"_view_.*$", r"_language_\d+.*$", r"_(?:table|tb)_\d+.*$", r"_light_.*$",
                r"_add_\d+.*$", r"_level\d+_sample\d+.*$"):
        name = re.sub(pat, "", name)
    return name


def select_cells(suite, n: int):
    """{(stem, family): [tid, ...] of length n} -- deterministic per cell, independent of the seed group
    (the same variants are evaluated for s01/s23/s45, only the policy differs)."""
    by = defaultdict(list)
    for tid, t in enumerate(suite.tasks):
        fam = classify(t.name)
        if fam in FAMILIES:
            by[(base_stem(t.name), fam)].append(tid)
    out = {}
    for key in sorted(by):
        ids = sorted(by[key])
        rng = random.Random(zlib.crc32("|".join(key).encode()))
        out[key] = sorted(rng.sample(ids, n)) if len(ids) >= n else [ids[i % len(ids)] for i in range(n)]
    return out


def clean_language(suite, tid: int, stem: str) -> str:
    from libero.libero import get_libero_path
    from libero.libero.envs import bddl_utils as BDDLUtils

    name = suite.tasks[tid].name
    if "_language_" in name:
        return suite.tasks[tid].language
    root = os.path.join(get_libero_path("bddl_files"), SUITE)
    for cand in (os.path.join(root, name + ".bddl"), os.path.join(root, stem + ".bddl")):
        if os.path.exists(cand):
            return BDDLUtils.get_problem_info(cand)["language_instruction"]
    return suite.tasks[tid].language


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--vanilla_task_map", required=True,
                    help="JSON {vanilla task id: base task name} -- the taskNN numbering stage 3 expects")
    ap.add_argument("--seed", type=int, default=20118)
    ap.add_argument("--n_per_cell", type=int, default=30)
    ap.add_argument("--num_uncertainty_sequences", type=int, default=16)
    ap.add_argument("--n_action_steps", type=int, default=0,
                    help="override the checkpoint's action execution horizon (0 = keep it). This study uses\n"
                         "HALF the action chunk: SmolVLA 50 -> 25 and X-VLA 30 -> 15 are the checkpoint\n"
                         "defaults, but FastWAM's default executes the full chunk (32), so it needs 16 here.")
    ap.add_argument("--max_steps", type=int, default=520)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--tasks", default="",
                    help="comma-separated VANILLA task numbers (00..09) to record; empty = all ten. Lets a\n"
                         "caller interleave record -> score -> delete per task when disk cannot hold the\n"
                         "whole 1,800-rollout tree (local runs).")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N rollouts")
    args = ap.parse_args()
    families = tuple(f for f in args.families.split(",") if f)

    _tl = torch.load  # the fork's init-state pickles need weights_only=False under torch >= 2.6
    torch.load = lambda *a, **k: _tl(*a, **{**k, "weights_only": False})

    from libero.libero import benchmark as lp_benchmark
    import lerobot.envs.libero as LE
    from lerobot.envs import configs as env_configs
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.fiper_data_generator.configuration_fiper_rollout_recorder import FiperRolloutRecorderConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.scripts.fiper_data_generation.record_fiper_rollout import rollout
    from lerobot.utils.random_utils import set_seed

    LE.get_task_init_states = lambda task_suite, i: task_suite.get_task_init_states(i)

    suite = lp_benchmark.get_benchmark_dict()[SUITE]()
    cells = select_cells(suite, args.n_per_cell)
    vanilla = {v: int(k) for k, v in json.load(open(args.vanilla_task_map)).items()}  # stem -> NN
    stems = sorted({k[0] for k in cells})
    missing = [s for s in stems if s not in vanilla]
    if missing:
        raise SystemExit(f"base tasks not in the vanilla map: {missing}")

    # Work units, sharded by cell so every shard mixes families.
    all_cells = [(s, f) for s in stems for f in families if (s, f) in cells]
    if args.tasks:
        want = {int(x) for x in args.tasks.split(",") if x.strip() != ""}
        all_cells = [(st, f) for st, f in all_cells if vanilla[st] in want]
        print(f"[shard {args.shard}] restricted to vanilla tasks {sorted(want)}: {len(all_cells)} cells", flush=True)
    my_cells = [c for i, c in enumerate(all_cells) if i % args.num_shards == args.shard]
    print(f"[shard {args.shard}/{args.num_shards}] {len(my_cells)} cells x {args.n_per_cell} rollouts", flush=True)

    set_seed(args.seed)
    policy_type = json.load(open(Path(args.policy_path) / "config.json"))["type"]
    policy = get_policy_class(policy_type).from_pretrained(pretrained_name_or_path=args.policy_path)
    if args.n_action_steps:
        policy.config.n_action_steps = args.n_action_steps
        print(f"[record] action execution horizon overridden to {args.n_action_steps} "
              f"(chunk {getattr(policy.config, 'chunk_size', None) or getattr(policy.config, 'action_horizon', None)})", flush=True)
    policy.eval()
    policy.to("cuda")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    env_cfg = env_configs.LiberoEnv(task=SUITE)
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    policy.init_fiper_rollout_recorder(config=FiperRolloutRecorderConfig(
        num_uncertainty_sequences=args.num_uncertainty_sequences,
        ode_eval_times=ODE_EVAL_TIMES,
        record_composed_inter_vel_diff=False,
    ))
    run_metadata = {
        "metadata": True, "task": SUITE,
        "action_prediction_horizon": policy.config.chunk_size,
        "action_execution_horizon": policy.config.n_action_steps,
        "action_batch_size": args.num_uncertainty_sequences,
        "benchmark": "libero_plus",
    }
    gym_kwargs = {k: v for k, v in dict(env_cfg.gym_kwargs).items() if k != "task_ids"}
    out_root = Path(args.output_dir)
    n_done = 0
    for stem, fam in my_cells:
        nn = vanilla[stem]
        test_dir = out_root / SUITE / f"task{nn:02d}" / "rollouts" / "test"
        fam_idx = FAMILIES.index(fam)
        for j, tid in enumerate(cells[(stem, fam)]):
            ep_idx = fam_idx * args.n_per_cell + j  # unique within the task dir across families
            if list(test_dir.glob(f"episode_?_{ep_idx:04d}_task{nn:02d}.pkl")):
                continue  # resume
            t0 = time.time()
            env = LE.LiberoEnv(task_suite=suite, task_id=tid, task_suite_name=SUITE, episode_length=args.max_steps,
                               camera_name=env_cfg.camera_name, init_states=True, episode_index=j, **gym_kwargs)
            # _add_/_level variants ship a single init state (the fork reshapes it to (1, -1)); the recorder's
            # rollout() resets without an init_state_id option, so wrap the episode index here.
            if getattr(env, "_init_states", None) is not None and len(env._init_states) > 0:
                env._init_state_id = j % len(env._init_states)
            lang = clean_language(suite, tid, stem)
            env.task_description = lang
            policy.fiper_rollout_recorder.reset()
            seed = args.seed + tid * 97 + j
            try:
                info, _ = rollout(env=env, policy=policy, env_preprocessor=env_pre, env_postprocessor=env_post,
                                  preprocessor=preprocessor, postprocessor=postprocessor, seed=seed)
            except Exception as exc:  # one broken variant must not take the shard down
                print(f"[{args.shard}] task{nn:02d} {fam} ep{ep_idx:03d} tid={tid} FAILED: {type(exc).__name__}: {exc}"[:300],
                      flush=True)
                try:
                    env.close()
                except Exception:
                    pass
                continue
            env.close()
            ep_metadata = {
                **run_metadata, **info,
                "task_id": nn, "episode": ep_idx, "seed": seed,
                "rollout_type": "test", "rollout_subtype": "ood",
                "perturbation": fam, "variant": suite.tasks[tid].name, "libero_plus_task_id": tid,
                "instruction": lang, "init_state_id": j,
            }
            policy.fiper_rollout_recorder.save_data(output_dir=test_dir, episode_metadata=ep_metadata)
            n_done += 1
            print(f"[{args.shard}] task{nn:02d} {fam:10s} ep{ep_idx:03d} tid={tid:5d} "
                  f"{'OK' if info['successful'] else '. '} {time.time() - t0:.0f}s", flush=True)
            if args.limit and n_done >= args.limit:
                print("limit reached"); return
    print(f"[shard {args.shard}] done: {n_done} new rollouts", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Watch weighted-pools train/eval jobs and resubmit failed incomplete paths."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "iterative_fine_tuning"
STATE_PATH = ROOT / "slurm" / "weighted_pools_monitor_state.json"
LOG_PATH = ROOT / "slurm" / "weighted_pools_monitor.log"

ACTIVE_STATES = {
    "COMPLETING",
    "CONFIGURING",
    "PENDING",
    "REQUEUED",
    "RESIZING",
    "RUNNING",
    "SUSPENDED",
}
FAILED_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}

RUNS = {
    "uniform_leak3_weighted_pools_lr_constant_history05_steps2000_s01": {
        "config": None,
        "seed_a": None,
        "seed_b": None,
        "train_job": None,
        "eval_job": "64672662",
    },
    "uniform_leak3_weighted_pools_lr_constant_history05_steps2000_s23": {
        "config": None,
        "seed_a": None,
        "seed_b": None,
        "train_job": None,
        "eval_job": "64672673",
    },
    "uniform_leak3_weighted_pools_lr_constant_history05_steps2000_s45": {
        "config": None,
        "seed_a": None,
        "seed_b": None,
        "train_job": None,
        "eval_job": "64717910",
    },
    "uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s01": {
        "config": "configs/iterative_fine_tuning/calibration_experiment/uniform_leak3_weighted_pools_lr_schedule.yaml",
        "seed_a": "0",
        "seed_b": "1",
        "train_job": "64717911",
        "eval_job": "64717913",
    },
    "uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s23": {
        "config": None,
        "seed_a": None,
        "seed_b": None,
        "train_job": None,
        "eval_job": "64672688",
    },
    "uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s45": {
        "config": "configs/iterative_fine_tuning/calibration_experiment/uniform_leak3_weighted_pools_lr_schedule.yaml",
        "seed_a": "4",
        "seed_b": "5",
        "train_job": "64717912",
        "eval_job": "64717914",
    },
}


def log(message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def normalize_state(raw: str | None) -> str | None:
    if raw is None:
        return None
    state = raw.strip().upper()
    if not state:
        return None
    return state.split()[0]


def job_state(job_id: str | None) -> str | None:
    if not job_id:
        return None
    sq = run_cmd(["squeue", "-h", "-j", str(job_id), "-o", "%T"])
    states = [normalize_state(line) for line in sq.stdout.splitlines() if line.strip()]
    states = [state for state in states if state]
    if states:
        return states[0]

    sacct = run_cmd(["sacct", "-n", "-j", str(job_id), "--format=State", "-P"])
    states = [normalize_state(line.split("|", 1)[0]) for line in sacct.stdout.splitlines() if line.strip()]
    states = [state for state in states if state]
    if not states:
        return None
    for state in states:
        if state in ACTIVE_STATES:
            return state
    for state in states:
        if state in FAILED_STATES:
            return state
    if "COMPLETED" in states:
        return "COMPLETED"
    return states[0]


def run_dir(run_name: str) -> Path:
    return OUT_ROOT / run_name


def train_rounds(run_name: str) -> int:
    return len(list(run_dir(run_name).glob("round_*/training_manifest.json")))


def eval_rounds(run_name: str) -> int:
    return len(list(run_dir(run_name).glob("round_*/round_evaluation.json")))


def is_active(job_id: str | None) -> bool:
    return job_state(job_id) in ACTIVE_STATES


def load_state() -> dict[str, dict[str, str | None]]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {
        run_name: {
            "train_job": meta.get("train_job"),
            "eval_job": meta.get("eval_job"),
        }
        for run_name, meta in RUNS.items()
    }


def save_state(state: dict[str, dict[str, str | None]]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def submit_eval(run_name: str) -> str | None:
    result = run_cmd(["sbatch", "--parsable", "cluster_eval.sbatch", run_name])
    if result.returncode != 0:
        log(f"{run_name}: eval submit failed: {result.stderr.strip()}")
        return None
    job_id = result.stdout.strip().split(";", 1)[0]
    log(f"{run_name}: submitted eval {job_id}")
    return job_id


def submit_train_and_eval(run_name: str, meta: dict[str, str | None]) -> tuple[str | None, str | None]:
    result = run_cmd(
        [
            "bash",
            "cluster_all.sh",
            str(meta["config"]),
            str(meta["seed_a"]),
            str(meta["seed_b"]),
            "--replay.history_fraction=0.5",
            "--replay.new_fraction=0.5",
            "--steps=2000",
            f"--paths.run_name={run_name}",
            f"--job_name={run_name}",
        ]
    )
    if result.returncode != 0:
        log(f"{run_name}: train/eval submit failed: {result.stderr.strip()}")
        return None, None

    train_job = None
    eval_job = None
    for line in result.stdout.splitlines():
        if "Train job submitted:" in line:
            train_job = line.rsplit(":", 1)[-1].strip()
        elif "Eval job submitted:" in line:
            eval_job = line.split("Eval job submitted:", 1)[-1].split("(", 1)[0].strip()
    log(f"{run_name}: submitted train {train_job} eval {eval_job}")
    return train_job, eval_job


def main() -> None:
    interval_s = int(float(__import__("os").environ.get("WEIGHTED_MONITOR_INTERVAL_S", "600")))
    duration_h = float(__import__("os").environ.get("WEIGHTED_MONITOR_DURATION_H", "14"))
    end_time = datetime.now() + timedelta(hours=duration_h)
    state = load_state()
    save_state(state)
    log(f"monitor started, interval={interval_s}s, until={end_time.isoformat(timespec='seconds')}")

    while datetime.now() < end_time:
        all_done = True
        for run_name, meta in RUNS.items():
            state.setdefault(run_name, {})
            trained = train_rounds(run_name)
            evaluated = eval_rounds(run_name)
            train_job = state[run_name].get("train_job")
            eval_job = state[run_name].get("eval_job")
            train_state = job_state(train_job)
            eval_state = job_state(eval_job)
            log(
                f"{run_name}: train={trained}/20 eval={evaluated}/20 "
                f"train_job={train_job}:{train_state} eval_job={eval_job}:{eval_state}"
            )

            if trained < 20 and meta.get("config") is not None:
                all_done = False
                if not is_active(train_job):
                    new_train, new_eval = submit_train_and_eval(run_name, meta)
                    if new_train:
                        state[run_name]["train_job"] = new_train
                    if new_eval:
                        state[run_name]["eval_job"] = new_eval
                    save_state(state)
                continue

            if evaluated < 20:
                all_done = False
                if not is_active(eval_job):
                    new_eval = submit_eval(run_name)
                    if new_eval:
                        state[run_name]["eval_job"] = new_eval
                        save_state(state)

        if all_done:
            log("all weighted-pools runs have 20 train rounds and 20 eval rounds; monitor exiting")
            return
        time.sleep(interval_s)

    log("monitor duration elapsed; exiting")


if __name__ == "__main__":
    main()

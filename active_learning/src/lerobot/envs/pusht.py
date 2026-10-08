"""Minimal Push-T benchmark helpers, mirroring the task-instruction API of
`lerobot.envs.libero` so Push-T can be plugged into
the iterative fine-tuning / active-learning pipeline as a single-task benchmark.

Push-T is single-task: one task group ("pusht") with one task id (0). The
instruction string matches the `lerobot/pusht` dataset task so the policy is
conditioned identically during pretraining, active fine-tuning, and evaluation.
"""

from __future__ import annotations

# Must match the task string in the lerobot/pusht dataset (meta.tasks).
PUSHT_INSTRUCTION = "Push the T-shaped block onto the T-shaped target."

# Single task group / id, used wherever the pipeline expects a {group: [ids]} map.
PUSHT_TASKS: dict[str, list[int]] = {"pusht": [0]}


def get_task_instruction(task_group: str = "pusht", task_id: int = 0) -> str:
    """Return the (constant) Push-T language instruction.

    Signature mirrors `lerobot.envs.libero.get_task_instruction`; the arguments
    are accepted for interface compatibility but Push-T has a single task.
    """
    return PUSHT_INSTRUCTION

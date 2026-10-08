from pathlib import Path

import pytest

from iterative_fine_tuning.config import (
    EnsembleTrainingConfig,
    IterativeFineTuningConfig,
    ReplayMixingConfig,
    SelectionConfig,
)
from iterative_fine_tuning.training import _normalize_worker_device
from lerobot.configs.default import DatasetConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


def test_parallel_member_training_requires_member_devices():
    cfg = EnsembleTrainingConfig(parallel_members=True, member_devices=[])

    with pytest.raises(ValueError, match="member_devices"):
        cfg.validate()


def test_resolved_member_devices_returns_requested_slice():
    cfg = IterativeFineTuningConfig(
        dataset=DatasetConfig(repo_id="HuggingFaceVLA/libero"),
        ensemble_training=EnsembleTrainingConfig(
            parallel_members=True,
            member_devices=["cuda:0", "cuda:1", "cuda:2"],
        ),
    )

    assert cfg.resolved_member_devices(2) == ["cuda:0", "cuda:1"]


def test_normalize_worker_device_uses_visible_gpu_index_for_cuda_devices():
    assert _normalize_worker_device("cuda:1") == ("cuda", "1")
    assert _normalize_worker_device("cuda") == ("cuda", None)
    assert _normalize_worker_device("cpu") == ("cpu", None)


def test_to_train_config_uses_fixed_learning_rate_without_scheduler():
    cfg = IterativeFineTuningConfig(
        dataset=DatasetConfig(repo_id="HuggingFaceVLA/libero"),
        use_policy_training_preset=False,
        optimizer=AdamWConfig(lr=1e-4, weight_decay=1e-10, grad_clip_norm=10.0),
        scheduler=CosineDecayWithWarmupSchedulerConfig(
            peak_lr=1e-4,
            decay_lr=2.5e-6,
            num_warmup_steps=100,
            num_decay_steps=1000,
        ),
        fixed_learning_rate=2.5e-5,
    )

    train_cfg = cfg.to_train_config(
        output_dir=Path("outputs/test_fixed_lr"),
        round_index=0,
        policy_cfg=SmolVLAConfig(),
    )

    assert train_cfg.optimizer is not None
    assert train_cfg.optimizer.lr == pytest.approx(2.5e-5)
    assert train_cfg.scheduler is None


def test_validate_rejects_non_positive_fixed_learning_rate():
    cfg = IterativeFineTuningConfig(
        dataset=DatasetConfig(
            repo_id="HuggingFaceVLA/libero",
            libero_tasks={"libero_10": [0]},
        ),
        policy=SmolVLAConfig(pretrained_path=Path("tmp/model")),
        replay=ReplayMixingConfig(history_tasks={"libero_10": [0]}),
        selection=SelectionConfig(
            ensemble_model_paths=["tmp/model", "tmp/model_2"],
        ),
        fixed_learning_rate=0.0,
    )

    with pytest.raises(ValueError, match="fixed_learning_rate"):
        cfg.validate()


def test_member_worker_command_uses_accelerate_launch_for_multi_gpu_devices(tmp_path):
    import sys

    from iterative_fine_tuning.training import _normalize_worker_device, member_worker_command

    request = tmp_path / "request.json"
    single = member_worker_command(request, _normalize_worker_device("cuda:1")[1])
    assert single == [sys.executable, "-m", "iterative_fine_tuning.train_worker", "--request_path", str(request)]
    assert member_worker_command(request, None) == single

    _, visible = _normalize_worker_device("cuda:0,1,2,3")
    assert visible == "0,1,2,3"
    multi = member_worker_command(request, visible)
    assert multi[:3] == [sys.executable, "-m", "accelerate.commands.launch"]
    assert multi[multi.index("--num_processes") + 1] == "4"
    assert multi[multi.index("--num_machines") + 1] == "1"
    port = int(multi[multi.index("--main_process_port") + 1])
    assert 1024 < port < 65536
    assert multi[-4:] == ["-m", "iterative_fine_tuning.train_worker", "--request_path", str(request)]

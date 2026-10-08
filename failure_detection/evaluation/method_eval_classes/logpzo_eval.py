"""logpZO failure detector (Xu et al., "Can We Detect Failures Without Failure Data?").

Ported to mirror the authors' reference implementation
(https://github.com/CXU-TRI/FAIL-Detect, `UQ_baselines/logpZO/train.py`,
`UQ_baselines/CFM/net_CFM.py`, `UQ_baselines/data_loader.py::adjust_xshape`,
`UQ_test/eval_load_baseline.py::logpZO_UQ`) as closely as the FIPER protocol allows:

* network      -- the diffusion_policy ConditionalUnet1D (Conv1d + GroupNorm + Mish residual
                  blocks, strided Downsample1d / ConvTranspose Upsample1d), no local or
                  global conditioning, dsed 128, down_dims [256, 512, 1024], kernel 5, 8 groups;
* input layout -- the flat embedding is zero-padded to a multiple of `in_dim` and reshaped to a
                  (L, in_dim) sequence with L a multiple of 4, exactly as `adjust_xshape` does,
                  so the 1-D UNet sees "channels x length" rather than a flat vector;
* time         -- t ~ U(0, 1) during training but the network is fed (t * 100).long(), i.e.
                  100 discrete steps into a sinusoidal embedding built for integers;
* training     -- Adam 1e-4, batch 128, fixed epoch budget, the *last* weights are used
                  (no validation split, no early stopping);
* score        -- one Euler step to the noise, z = x + v(x, t=0), score = ||z||^2 summed over
                  every dim including the padding.

The training data is a config switch (`training_data`):

* "demos"       -- the policy's own training demonstrations, embedded offline by
                   active_learning/scripts/failure_detection/run_{pusht,libero_plus}.sh into
                   `<task_data_path>/<demo_embeddings_file>`. This is the reference's setting.
                   Because nothing is then fit on calibration rollouts, the detector keeps the
                   full calibration set for thresholding (trains_on_calibration_data = False).
* "calibration" -- FIPER's calibration-only protocol: fit on the training half of the
                   calibration rollouts (shared_utils.calibration_split), threshold on the rest.
                   Here an epoch is only 1-5 gradient steps, so the reference's epoch budget is
                   not comparable.
"""

from .base_eval_class import BaseEvalClass
import copy
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from shared_utils.calibration_split import get_calibration_split
from shared_utils.model_checkpoints import find_checkpoint, save_checkpoint
from shared_utils.training_data import (
    demo_training_identity,
    demo_training_tensor,
    load_demo_embeddings,
    resolve_training_data,
    is_pooled,
    is_demo_pooled,
    uses_pooled_models_dir,
    demo_aligner,
    pooled_aligner,
    pooled_models_dir,
    pooled_training_tensors,
    trains_on_calibration_data,
)
from shared_utils.nn_base.small_submodules import SinusoidalPosEmb
from typing import Optional, Union


def adjust_xshape(x: torch.Tensor, in_dim: int) -> torch.Tensor:
    """Zero-pad a flat embedding batch and reshape it to a (B, L, in_dim) sequence.

    Exact port of FAIL-Detect `UQ_baselines/data_loader.py::adjust_xshape`. Two padding
    passes: first so the total width is a multiple of `in_dim`, then so the resulting
    sequence length L is a multiple of 4 (the UNet halves L twice, so the up path must be
    able to double it back to the same size).
    """
    total_dim = x.shape[1]
    remain_dim = total_dim % in_dim
    if remain_dim > 0:
        pad = in_dim - remain_dim
        total_dim += pad
        x = torch.cat([x, torch.zeros(x.shape[0], pad, device=x.device, dtype=x.dtype)], dim=1)
    reshaped_dim = total_dim // in_dim
    if reshaped_dim % 4 != 0:
        extra_pad = (4 - (reshaped_dim % 4)) * in_dim
        x = torch.cat([x, torch.zeros(x.shape[0], extra_pad, device=x.device, dtype=x.dtype)], dim=1)
    return x.reshape(x.shape[0], -1, in_dim)


class LOGPZOEval(BaseEvalClass):
    """logpZO: a flow-matching model of the calibration observation embeddings.

    The uncertainty score is the squared norm of the noise an embedding maps to under one
    backward Euler step, so embeddings far from the calibration distribution score high.
    The model is fit on the *training half* of the calibration episodes only, so the
    thresholds the base class then computes on the held-out half are out-of-sample. With
    `logpzo_train.cache` the fitted model is stored under `<task_data_path>/logpzo_models/`
    and reused instead of refitted on later runs.
    """

    def __init__(self, cfg, method_name, device, task_data_path, dataset, **kwargs):
        self.training_data = resolve_training_data(cfg)
        # Only the calibration-trained variant has in-sample calibration scores to hold out.
        self.trains_on_calibration_data = trains_on_calibration_data(cfg)
        # training_data=calibration_pooled: one model per policy, fit on the union of the training
        # halves of all pooled tasks (pipeline.py passes the datasets) and stored in a shared dir.
        self.pooled = is_pooled(cfg)
        self.pooled_datasets = kwargs.pop("pooled_datasets", None)
        self.pooled_models_dir = pooled_models_dir(kwargs, task_data_path)
        kwargs.pop("pooled_models_dir", None)
        self._pooled_cache = None
        # Embedding alignment across tasks (SmolVLA: instruction-length-dependent width), pooled mode only.
        # training_data=demos_pooled: one model per policy fit on the pre-padded multi-task demo
        # embeddings; each task's rollouts are padded to that width with the same rule.
        self.demo_pooled = is_demo_pooled(cfg)
        if self.pooled:
            self.aligner = pooled_aligner(cfg, self.pooled_datasets)
        elif self.demo_pooled:
            self.aligner = demo_aligner(cfg, load_demo_embeddings(task_data_path, cfg))
        else:
            self.aligner = None
        super().__init__(cfg, method_name, device, task_data_path, dataset, **kwargs)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedding_dim = int(self.dataset.get_tensor_shape("obs_embeddings")[-1])
        if self.aligner is not None:
            self.embedding_dim = self.aligner.target_dim
        if uses_pooled_models_dir(cfg):
            self.checkpoint_dir = os.path.join(self.pooled_models_dir, "logpzo_models", self.method_name)
        else:
            self.checkpoint_dir = os.path.join(self.task_data_path, "logpzo_models", self.method_name)

        # Reference: in_dim is the channel width the flat embedding is folded into
        # (10 for square/can, 20 for transport/tool_hang). It is a hyperparameter, not a
        # property of the embedding, so it lives in the config.
        self.in_dim = int(self.cfg.in_dim)
        self.time_scale = float(self.cfg.time_scale)

        # Seed BEFORE constructing the model: the parameter initialisation draws from the
        # global RNG, so seeding afterwards would leave the init unseeded and make logpZO
        # irreproducible despite eval.deterministic.
        if self.cfg.get("deterministic", False):
            torch.manual_seed(int(self.cfg.get("seed", 0)))

        self.model = ConditionalUnet1D(
            input_dim=self.in_dim,
            diffusion_step_embed_dim=int(self.cfg.diffusion_step_embed_dim),
            down_dims=list(self.cfg.down_dims),
            kernel_size=int(self.cfg.kernel_size),
            n_groups=int(self.cfg.n_groups),
            cond_predict_scale=bool(self.cfg.cond_predict_scale),
        )
        self.model.to(self.device)
        self.trainer = LogpZOTrainer(
            model=self.model,
            num_epochs=int(self._per_mode(self.cfg.num_epochs)),
            batch_size=int(self.cfg.batch_size),
            learning_rate=float(self.cfg.learning_rate),
            time_scale=self.time_scale,
            model_selection=str(self.cfg.model_selection),
            micro_batch_size=self.cfg.get("micro_batch_size", None),
        )

    def _per_mode(self, value):
        """Config values may be given per training mode ({demos: .., calibration: ..}) or as a scalar."""
        if isinstance(value, (dict, DictConfig)):
            # demos_pooled shares the demos entries (same data regime, ~30k demo frames). Falling back
            # instead of adding a key keeps `hparams.model.num_epochs` -- and with it the identity of every
            # cached per-task demo checkpoint -- unchanged.
            if self.training_data not in value and self.training_data == "demos_pooled":
                return value["demos"]
            return value[self.training_data]
        return value

    def _checkpoint_hparams(self) -> dict:
        """Hyperparameters identifying a cached model.

        Everything that changes the fitted weights belongs here, so that altering any of
        it retrains instead of silently reusing a stale checkpoint. That includes the
        calibration split: a model fit on a different half of the episodes is a
        different model.

        The *number* of training episodes must be recorded too, not just the split
        fraction. Evaluating the same recording at different calibration-set sizes (see
        `max_calibration_rollouts`) leaves train_fraction unchanged while halving or
        doubling the data the model actually sees -- without this key those runs would
        collide on one checkpoint and silently reuse the wrong weights.
        """
        training = {
            "embedding_dim": self.embedding_dim,
            "deterministic": bool(self.cfg.get("deterministic", False)),
            "seed": int(self.cfg.get("seed", 0)),
        }
        if self.training_data in ("demos", "demos_pooled"):
            training.update(demo_training_identity(load_demo_embeddings(self.task_data_path, self.cfg)))
            if self.aligner is not None:
                training.update(self.aligner.identity())
        elif self.pooled:
            training.update(self._pooled()[1])
        else:
            train_fraction, split_seed = get_calibration_split(self.cfg)
            training.update(
                calibration_train_fraction=train_fraction,
                calibration_split_seed=split_seed,
                num_train_episodes=len(self._calibration_train_episode_indices()),
            )
        return {"model": OmegaConf.to_container(self.cfg.hparams.model, resolve=True), "training": training}

    def _execute_preprocessing(self):
        """Load the cached flow-matching model, or fit one and cache it."""
        hparams = self._checkpoint_hparams()
        train_cfg = self.cfg.get("logpzo_train", {}) or {}
        cache = bool(self._per_mode(train_cfg.get("cache", True)))
        overwrite = bool(train_cfg.get("overwrite", False))

        if cache and not overwrite:
            found = find_checkpoint(self.checkpoint_dir, desired_hparams=hparams)
            if found is not None:
                _, checkpoint = found
                self.model.load_state_dict(checkpoint["state_dict"])
                self.model.to(self.device)
                self.model.eval()
                print(f"[logpzo] loaded cached model from {self.checkpoint_dir}")
                return

        # Train a flow matching model to learn the distribution of the embeddings
        embeddings = adjust_xshape(self._get_training_embeddings(), self.in_dim)
        self.trainer.train(embeddings=embeddings)
        self.model.load_state_dict(self.trainer.selected_state_dict)
        self.model.eval()

        if cache:
            save_checkpoint(
                self.checkpoint_dir,
                state_dict=self.model.state_dict(),
                model_cfg={"hparams": hparams, "input_dict": {"input_dim": self.in_dim}},
                checkpoint_name=f"checkpoint_{self.method_name}.ckpt",
                overwrite=True,
            )

    def _get_training_embeddings(self) -> torch.Tensor:
        """Observation embeddings of the calibration episodes reserved for training.

        Iterating episode-wise (rather than `get_subset`, which concatenates the whole
        calibration set) is what makes the episode-level split possible; the enumeration
        order matches the one the base class uses for thresholding.
        """
        if self.training_data in ("demos", "demos_pooled"):
            return demo_training_tensor(self.task_data_path, self.cfg, self.embedding_dim, self.device, "logpzo")
        if self.pooled:
            return self._pooled()[0]["obs_embeddings"].to(self.device)

        train_indices = self._calibration_train_episode_indices()
        episode_embeddings = []
        for episode_idx, episode_data in enumerate(
            self.dataset.iterate_episodes(
                subset="calibration",
                required_tensors=["obs_embeddings"],
                normalize_tensors=self.normalize_tensors,
                history=self.cfg.history_length,
            )
        ):
            if episode_idx not in train_indices:
                continue
            episode_embeddings.append(episode_data["obs_embeddings"])

        if not episode_embeddings:
            raise ValueError("logpZO calibration training split produced no training episodes.")

        return torch.cat(episode_embeddings, dim=0).to(self.device)

    def _pooled(self):
        """(tensors, identity) of the pooled training set, computed once per evaluator."""
        if self._pooled_cache is None:
            self._pooled_cache = pooled_training_tensors(
                self.pooled_datasets, self.cfg, required_tensors=["obs_embeddings"],
                history=self.cfg.history_length, normalize_tensors=self.normalize_tensors, tag="logpzo",
            )
        return self._pooled_cache

    def load_model(self):
        # The model is built in __init__ and fitted/loaded in _execute_preprocessing.
        pass

    def calculate_uncertainty_score(self, rollout_tensor_dict, **kwargs):
        """Reference `logpZO_UQ`: z = x + v(x, t=0); score = ||z||^2 over all dims."""
        obs_embeddings = rollout_tensor_dict["obs_embeddings"]
        if len(obs_embeddings.shape) > 1:
            obs_embeddings = obs_embeddings[0]
        if self.aligner is not None:
            obs_embeddings = self.aligner(obs_embeddings)
        obs_embeddings = adjust_xshape(obs_embeddings.unsqueeze(0).to(self.device), self.in_dim)

        with torch.no_grad():
            timesteps = torch.zeros(obs_embeddings.shape[0], dtype=torch.long, device=self.device)
            pred_v = self.model(obs_embeddings, timesteps)
            noise = obs_embeddings + pred_v
            uncertainty_score = noise.reshape(noise.shape[0], -1).pow(2).sum(dim=-1).item()

        return uncertainty_score


class LogpZOTrainer:
    """Reference `UQ_baselines/logpZO/train.py` training loop.

    Conditional flow matching from data x0 to Gaussian noise x1 along the straight
    interpolant x_t = x0 + t (x1 - x0), regressing the constant velocity x1 - x0 with an MSE
    loss. The continuous time t ~ U(0, 1) is fed to the network as (t * time_scale).long():
    the UNet's sinusoidal embedding was designed for integer diffusion steps and is nearly
    constant over [0, 1], so without the scaling the model is effectively time-unconditional.

    `model_selection="last"` mirrors the reference (fixed epoch budget, no validation split,
    final weights). `"best_val"` re-enables the 90/10 split and best-validation snapshot of
    the earlier FIPER implementation.
    """

    def __init__(self, model, num_epochs, batch_size, learning_rate, time_scale=100.0, model_selection="last",
                 micro_batch_size=None):
        if model_selection not in ("last", "best_val"):
            raise ValueError(f"model_selection must be 'last' or 'best_val', got {model_selection!r}")
        self.model = model
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        # Gradient accumulation: each optimizer step still uses `batch_size` samples, processed in
        # chunks of `micro_batch_size` to bound activation memory. Equivalent to the un-chunked
        # update (the loss is a per-sample mean and GroupNorm normalises per sample), up to
        # floating-point summation order. None = no chunking.
        self.micro_batch_size = int(micro_batch_size) if micro_batch_size else None
        self.learning_rate = learning_rate
        self.time_scale = time_scale
        self.model_selection = model_selection
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Diagnostics and the weights chosen by model_selection.
        self.selected_state_dict = None
        self.best_epoch = None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def _flow_matching_loss(self, batch: torch.Tensor) -> torch.Tensor:
        x0, x1 = batch, torch.randn_like(batch)
        vtrue = x1 - x0
        cont_t = torch.rand(batch.shape[0], device=batch.device)
        xnow = x0 + cont_t.view(-1, *([1] * (batch.ndim - 1))) * vtrue
        vhat = self.model(xnow, (cont_t * self.time_scale).long())
        return (vhat - vtrue).pow(2).mean()

    def train(self, embeddings: torch.Tensor):
        if self.model_selection == "best_val":
            train_size = int(0.9 * len(embeddings))
            train_dataset, val_dataset = torch.utils.data.random_split(
                embeddings, [train_size, len(embeddings) - train_size]
            )
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            train_dataset, val_loader = embeddings, None
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        self.train_losses, self.val_losses = [], []
        # Only draw the bar on a terminal -- redirected into a log file the per-epoch
        # updates add megabytes of carriage returns per pipeline run.
        progress_bar = tqdm(range(self.num_epochs), desc="Training logpZO", leave=False, disable=not sys.stderr.isatty())
        for epoch in progress_bar:
            self.model.train()
            epoch_losses = []
            for batch in train_loader:
                self.optimizer.zero_grad()
                n = batch.shape[0]
                mb = self.micro_batch_size or n
                batch_loss = 0.0
                for start in range(0, n, mb):
                    chunk = batch[start:start + mb]
                    # weight so the accumulated gradient equals that of the full-batch mean loss
                    loss = self._flow_matching_loss(chunk) * (chunk.shape[0] / n)
                    if torch.isnan(loss):
                        raise ValueError(f"logpZO loss is NaN at epoch {epoch + 1}")
                    loss.backward()
                    batch_loss += loss.item()
                self.optimizer.step()
                epoch_losses.append(batch_loss)
            epoch_loss = float(np.mean(epoch_losses))
            self.train_losses.append(epoch_loss)

            val_loss = float("nan")
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = float(np.mean([self._flow_matching_loss(b).item() for b in val_loader]))
                self.val_losses.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Deep-copy: state_dict() returns references to the live parameters,
                    # which the optimizer keeps mutating in place.
                    self.selected_state_dict = copy.deepcopy(self.model.state_dict())
                    self.best_epoch = epoch + 1

            progress_bar.set_description(f"logpZO epoch {epoch + 1}: train {epoch_loss:.4f} val {val_loss:.4f}")
            # Bar suppressed (non-tty): log four times per fit regardless of the epoch budget.
            log_every = max(1, self.num_epochs // 4)
            if progress_bar.disable and ((epoch + 1) % log_every == 0 or epoch + 1 == self.num_epochs):
                print(f"[logpzo] epoch {epoch + 1}/{self.num_epochs} train {epoch_loss:.4f} val {val_loss:.4f}", flush=True)

        if self.model_selection == "last":
            self.selected_state_dict = copy.deepcopy(self.model.state_dict())
            self.best_epoch = self.num_epochs


# ---------------------------------------------------------------------------------------
# Reference network: diffusion_policy.model.diffusion.conditional_unet1d.ConditionalUnet1D
# with local_cond_dim=None and global_cond_dim=None, plus its conv1d building blocks.
# Kept verbatim in structure (einops replaced by .transpose) so it can be diffed against
# the original.
# ---------------------------------------------------------------------------------------


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8, cond_predict_scale=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        # FiLM modulation: predicts per-channel scale and bias
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """x: (B, in_channels, L); cond: (B, cond_dim) -> (B, out_channels, L)"""
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)[:, :, None]  # Rearrange('batch t -> batch t 1')
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            out = embed[:, 0, ...] * out + embed[:, 1, ...]
        else:
            out = out + embed
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
        ])

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int], **kwargs):
        """sample: (B, L, input_dim); timestep: (B,) or scalar -> (B, L, input_dim)"""
        sample = sample.transpose(1, 2)  # 'b h t -> b t h'

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        global_feature = self.diffusion_step_encoder(timesteps)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.transpose(1, 2)  # 'b t h -> b h t'

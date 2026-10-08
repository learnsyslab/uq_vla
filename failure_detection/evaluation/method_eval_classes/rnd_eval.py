from .base_eval_class import BaseEvalClass
import numpy as np
import torch
import os
import importlib
import time
from rnd import RND_OE, RND_TO, RND_OI, RND_T, RND_TO_L, RNDBase
from hydra.utils import get_class
from omegaconf import OmegaConf
from shared_utils.training_data import (
    demo_training_identity,
    is_pooled,
    is_demo_trained,
    is_demo_pooled,
    uses_pooled_models_dir,
    demo_aligner,
    load_demo_embeddings,
    pooled_aligner,
    pooled_models_dir,
    pooled_training_tensors,
    resolve_training_data,
    trains_on_calibration_data,
)


class RNDEval(BaseEvalClass):
    """Class for evaluating all RND models."""

    def __init__(self, cfg, method_name, device, task_data_path, dataset, **kwargs):
        # Fit on calibration rollouts -> the base class holds the training half back from
        # thresholding. Fit on demos -> nothing to hold back, all calibration episodes are used.
        self.trains_on_calibration_data = trains_on_calibration_data(cfg)
        # training_data=calibration_pooled: the shared model dir and the task datasets it pooled
        # (needed only to recompute the checkpoint identity the trainer stored).
        self.pooled_models_dir = pooled_models_dir(kwargs, task_data_path)
        self.pooled_datasets = kwargs.pop("pooled_datasets", None)
        kwargs.pop("pooled_models_dir", None)
        # Embedding alignment across tasks (SmolVLA: instruction-length-dependent width), pooled mode only.
        if is_pooled(cfg):
            self.aligner = pooled_aligner(cfg, self.pooled_datasets)
        elif is_demo_pooled(cfg):
            # One model per policy on pre-padded multi-task demos; pad this task's rollouts alike.
            self.aligner = demo_aligner(cfg, load_demo_embeddings(task_data_path, cfg))
        else:
            self.aligner = None
        super().__init__(cfg, method_name, device, task_data_path, dataset, **kwargs)

    def load_model(self):
        """Load the RND model based on the method_name specified in the configuration file."""
        # Define the model class
        class_name = self.method_name
        if self.method_name.startswith("nrnd"):
            class_name.replace("nrnd", "rnd")
        class_name = f"{class_name.upper()}"
        # Load the model (with hydra)
        if uses_pooled_models_dir(self.cfg):
            checkpoint_dir = os.path.join(self.pooled_models_dir, "rnd_models", self.method_name)
        else:
            checkpoint_dir = os.path.join(self.task_data_path, "rnd_models", self.method_name)
        world_model_dir = os.path.join(self.task_data_path, "sys_id")
        # Same identity the trainer saved: hparams.model plus, for demo-trained models, which
        # demonstrations they were fit on. Must stay a DictConfig: RNDBase._load_checkpoint compares
        # node types strictly, and the stored hparams are DictConfig nodes (a plain dict would make
        # every list-valued key compare list vs ListConfig and fail).
        desired_hparams = OmegaConf.create(OmegaConf.to_container(self.cfg.hparams.model, resolve=True))
        if is_demo_trained(self.cfg):
            identity = demo_training_identity(load_demo_embeddings(self.task_data_path, self.cfg))
            if self.aligner is not None:
                identity.update(self.aligner.identity())
            desired_hparams["demo_training"] = identity
        elif is_pooled(self.cfg):
            _, identity = pooled_training_tensors(
                self.pooled_datasets, self.cfg,
                required_tensors=self.cfg.required_tensors, optional_tensors=self.cfg.optional_tensors,
                required_actions=self.cfg.required_actions, optional_actions=self.cfg.optional_actions,
                history=self.cfg.history_length, normalize_tensors=dict(self.cfg.normalize_tensors),
                tag=f"{self.method_name} identity",
            )
            desired_hparams["pooled_training"] = identity
        rnd_model: RNDBase = get_class(f"rnd.{class_name}")(
            checkpoint_dir=checkpoint_dir, desired_hparams=desired_hparams, world_model_dir=world_model_dir
        )
        rnd_model.to(self.device)
        rnd_model.eval()
        self.model = rnd_model

    def calculate_uncertainty_score(self, rollout_tensor_dict: dict, **kwargs):
        """Calculate the uncertainty score for a single step."""
        for key in rollout_tensor_dict.keys():
            tensor = rollout_tensor_dict[key]
            if key == "obs_embeddings" and self.aligner is not None:
                tensor = self.aligner(tensor)
            rollout_tensor_dict[key] = tensor.unsqueeze(0).to(self.device)

        uncertainty_scores = (
            self.model(**self.model.datasets_to_model_inputs(datasets=rollout_tensor_dict)).detach().cpu()
        )
        if len(uncertainty_scores) == 1:
            uncertainty_scores = uncertainty_scores.item()
        else:
            uncertainty_scores = uncertainty_scores.numpy()
        return uncertainty_scores

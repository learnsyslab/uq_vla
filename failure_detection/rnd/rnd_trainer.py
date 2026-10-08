import os
import shutil
import sys
import pathlib
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm
from rnd.rnd_models import RND_OE, RND_TO_L, RND_T, RND_OI, RND_TO, RNDBase
from rollout_data.rollout_datasets import ProcessedRolloutDataset
from omegaconf import DictConfig, OmegaConf
from shared_utils.hydra_utils import load_config
from shared_utils.utility_functions import ensure_list
from shared_utils.calibration_split import get_calibration_split_indices, splits_calibration_set
from shared_utils.training_data import (
    apply_method_cfg_overrides,
    demo_training_identity,
    demo_training_tensor,
    load_demo_embeddings,
    resolve_training_data,
    is_pooled,
    is_demo_trained,
    is_demo_pooled,
    demo_aligner,
    pooled_models_dir,
    pooled_training_tensors,
)
from typing import Union, Dict, List, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import hydra
import copy


class RNDTrainer:
    def __init__(
        self,
        base_config_path,
        task_data_path,
        dataset: ProcessedRolloutDataset,
        device=None,
        **kwargs,
    ):
        """Initialize the RND trainer.
        Args:
            task (str): The task for which the RND models are trained.
            policy (DiffusionPolicy): The policy used to generate the rollouts.
            rnd_models (list): The list of RND models to train.
            config_path (str): The path to the configuration file.
            overwrite (bool): Whether to overwrite existing training data.
        Purpose:
            The RND trainer is used to train the RND models for a given task.
            It generates the training datasets from the raw datasets and trains the specified RND models.
            The training datasets are saved in the "/data_all/{task}/rnd_training_data/" directory.
            The trained models are saved in the "/data_all/{task}/{model}/" directory.
            The only function that should be called from outside is the train function.
        """
        self.base_config_path = base_config_path
        self.task_data_path = task_data_path
        self.training_data_dir = os.path.join(self.task_data_path, "rnd_training_data")
        if device is not None and device in ["cpu", "cuda:0"]:
            self.device = device if torch.cuda.is_available() else "cpu"
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset: ProcessedRolloutDataset = dataset

        self.task_cfg = kwargs.get("task_cfg", None)
        # Experiment-level keys (training_data, demo_embeddings_file) forwarded by pipeline.py.
        self.method_cfg_overrides = kwargs.get("method_cfg_overrides", None)
        # training_data=calibration_pooled: all task datasets (pipeline.py) and the shared model dir.
        self.pooled_datasets = kwargs.get("pooled_datasets", None)
        self.pooled_models_dir = pooled_models_dir(kwargs, self.task_data_path)
        self._pooled_identity_cache: dict[str, dict] = {}

    def train(self, rnd_models):
        """Train the specified RND models."""
        for model in rnd_models:
            print(f"Training {model}")
            self._train_model(model)

    def _train_model(self, model_name):
        """Train the specified RND model."""
        # Load config and set seed
        filename = model_name
        if model_name.startswith("nrnd"):
            filename.replace("nrnd", "rnd")
        cfg = load_config(module="eval", filename=filename, return_only_subdict=True)
        apply_method_cfg_overrides(cfg, self.method_cfg_overrides, method_name=model_name)
        # Fix hparams to prevent input dict modifications in rnd class from changing them.
        # Mutate the plain dict before assigning it back: once assigned, OmegaConf re-wraps it as
        # a struct-mode node that refuses new keys.
        hparams = OmegaConf.to_container(cfg.hparams, resolve=True)
        if is_demo_trained(cfg):
            # Part of the checkpoint identity: RNDEval adds the same keys when loading.
            hparams["model"]["demo_training"] = demo_training_identity(load_demo_embeddings(self.task_data_path, cfg))
            aligner = self._demo_aligner(cfg)
            if aligner is not None:
                hparams["model"]["demo_training"].update(aligner.identity())
        pooled = is_pooled(cfg)
        if pooled:
            # One model per policy: identity = which tasks / episodes it pooled (RNDEval adds the same).
            hparams["model"]["pooled_training"] = self._pooled_identity(model_name, cfg)
        cfg.hparams = hparams
        if cfg.rnd_train.get("deterministic", False):
            self._set_seed(cfg.rnd_train.get("seed", None))

        # Define data path, load dataset, and save directory
        if pooled or is_demo_pooled(cfg):
            # Shared across tasks (demos_pooled: a per-policy cache that is never cleared).
            # calibration_pooled: pipeline.py clears it once per run, so `overwrite` must not wipe
            # the model the first task trained before the other tasks reuse it.
            save_dir = os.path.join(self.pooled_models_dir, "rnd_models", model_name)
        else:
            save_dir = os.path.join(self.task_data_path, "rnd_models", model_name)
            # If it exists, remove the directory -- except for a demo-trained model, which is cached
            # like logpZO's (`logpzo_train.cache.demos`): it does not depend on the calibration draw,
            # so one fit per task serves every calibration size / subset seed. `_check_for_existance`
            # still refits when the demo identity (episodes, frames, policy) or hparams differ.
            demo_cached = resolve_training_data(cfg) == "demos"
            if os.path.exists(save_dir) and cfg.rnd_train.get("overwrite", False) and not demo_cached:
                shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        # Normaliztion variant use the same training function as the non-normalization variant
        self._training_loop(
            save_dir,
            cfg,
            model_name=filename,
            world_model_dir=os.path.join(self.task_data_path, "sys_id"),
        )

    def _get_datasets_for_model(self, model_name: str, cfg: DictConfig, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Load and filter datasets based on the model's requirements.
        Args:
            model_name (str): The name of the model.
        Returns:
            dict: A dictionary of datasets.
        """
        required_datasets = cfg.required_tensors
        optional_datasets = cfg.optional_tensors

        if is_demo_trained(cfg):
            # Demo embeddings exist only for the observation embedding, stored raw (the extractor
            # saves the policy's global_cond as is), so the model must want exactly that, unnormalised.
            if list(required_datasets) != ["obs_embeddings"] or list(optional_datasets or []):
                raise NotImplementedError(
                    f"training_data=demos is only implemented for models whose sole input is "
                    f"obs_embeddings (got required={list(required_datasets)}, optional={list(optional_datasets or [])})."
                )
            if cfg.normalize_tensors.get("obs_embeddings", False):
                raise ValueError("training_data=demos requires normalize_tensors.obs_embeddings=False: "
                                 "the demo embeddings are stored raw and would not match the rollouts.")
            embedding_dim = int(self.dataset.get_tensor_shape("obs_embeddings")[-1])
            aligner = self._demo_aligner(cfg)
            if aligner is not None:
                embedding_dim = aligner.target_dim  # pooled demos are pre-padded to the common width
            return {"obs_embeddings": demo_training_tensor(self.task_data_path, cfg, embedding_dim, "cpu", model_name)}

        if is_pooled(cfg):
            tensors, _ = self._pooled_tensors(model_name, cfg)
            return tensors

        train_indices, _ = get_calibration_split_indices(self.dataset, cfg)
        use_episode_split = splits_calibration_set(cfg)

        if use_episode_split:
            episode_tensors_by_key: dict[str, list[torch.Tensor]] = {}
            for episode_idx, episode_data in enumerate(
                self.dataset.iterate_episodes(
                    subset="calibration",
                    required_tensors=required_datasets,
                    optional_tensors=optional_datasets,
                    required_actions=cfg.required_actions,
                    optional_actions=cfg.optional_actions,
                    history=cfg.history_length,
                    normalize_tensors=dict(cfg.normalize_tensors),
                )
            ):
                if episode_idx not in train_indices:
                    continue
                for key, tensor in episode_data.items():
                    episode_tensors_by_key.setdefault(key, []).append(tensor)

            if not episode_tensors_by_key:
                raise ValueError("RND calibration training split produced no training episodes.")

            return {
                key: torch.cat(tensors, dim=0)
                for key, tensors in episode_tensors_by_key.items()
            }

        # Returns all datasets requested but the optional ones if not available
        datasets = self.dataset.get_subset(
            subset="calibration",
            required_tensors=required_datasets,
            optional_tensors=optional_datasets,
            return_as_list=False,
            required_actions=cfg.required_actions,
            optional_actions=cfg.optional_actions,
            history=cfg.history_length,
            normalize_tensors=dict(cfg.normalize_tensors),
        )

        # Create a dictionary of datasets if as list
        if isinstance(datasets, list):
            dataset_dict = {
                name: dataset
                for name, dataset in zip(required_datasets + optional_datasets, datasets)
                if dataset is not None
            }
            return dataset_dict
        return datasets

    def _demo_aligner(self, cfg: DictConfig):
        """training_data=demos_pooled: pads this task's rollouts to the pooled demo width (None otherwise)."""
        if not is_demo_pooled(cfg):
            return None
        return demo_aligner(cfg, load_demo_embeddings(self.task_data_path, cfg))

    def _pooled_tensors(self, model_name: str, cfg: DictConfig):
        """Training tensors + identity for training_data=calibration_pooled (computed once per model)."""
        key = str(model_name).lower()
        if key not in self._pooled_identity_cache:
            tensors, identity = pooled_training_tensors(
                self.pooled_datasets, cfg,
                required_tensors=cfg.required_tensors, optional_tensors=cfg.optional_tensors,
                required_actions=cfg.required_actions, optional_actions=cfg.optional_actions,
                history=cfg.history_length, normalize_tensors=dict(cfg.normalize_tensors), tag=key,
            )
            self._pooled_identity_cache[key] = {"tensors": tensors, "identity": identity}
        entry = self._pooled_identity_cache[key]
        return entry["tensors"], entry["identity"]

    def _pooled_identity(self, model_name: str, cfg: DictConfig) -> dict:
        return self._pooled_tensors(model_name, cfg)[1]

    def _create_dataloader(self, dataset_dict: Dict[str, torch.Tensor], **kwargs) -> DataLoader:
        """
        Create a DataLoader from the dataset dictionary.
        Args:
            dataset_dict (dict): A dictionary of datasets.
            batch_size (int): The batch size.
        Returns:
            DataLoader: The DataLoader object.
        """
        # Combine datasets into a TensorDataset
        tensors = list(dataset_dict.values())
        dataset = TensorDataset(*tensors)
        dataloader = DataLoader(
            dataset,
            batch_size=kwargs.get("batch_size", 4),
            shuffle=kwargs.get("shuffle", True),
            num_workers=kwargs.get("num_workers", 4),
            pin_memory=kwargs.get("pin_memory", True),
        )
        return dataloader

    def _get_optimizers(self, model, cfg):
        if cfg.get("rnd_train", None) is not None:
            cfg = cfg.rnd_train
        if cfg.optimizer == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, eps=cfg.eps, weight_decay=cfg.weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.eps)

        # Add a cosine learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs, eta_min=cfg.get("lr_min", 1e-6))
        return optimizer, scheduler

    def _split_datasets(
        self, dataset_dict: Dict[str, torch.Tensor], train_ratio: float = 0.9
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Split the dataset dictionary into training and validation sets.

        Args:
            dataset_dict (dict): A dictionary of datasets where keys are dataset names and values are tensors.
            train_ratio (float): The ratio of the dataset to use for training. The rest is used for validation.

        Returns:
            Tuple[dict, dict]: Two dictionaries (train_dataset_dict, val_dataset_dict) containing the split datasets.
        """
        # Ensure the dataset dictionary is not empty
        if not dataset_dict:
            raise ValueError("The dataset dictionary is empty.")

        # Get the length of the datasets (assumes all datasets have the same length)
        dataset_length = dataset_dict[list(dataset_dict.keys())[0]].shape[0]

        # Generate train and validation indices
        val_indices = np.random.choice(dataset_length, int(dataset_length * (1 - train_ratio)), replace=False)
        train_indices = np.setdiff1d(np.arange(dataset_length), val_indices)

        # Split each dataset in the dictionary
        train_dataset_dict = {key: dataset[train_indices] for key, dataset in dataset_dict.items()}
        val_dataset_dict = {key: dataset[val_indices] for key, dataset in dataset_dict.items()}

        return train_dataset_dict, val_dataset_dict

    def _init_rnd_model(
        self,
        model_name: str,
        cfg: DictConfig,
        **kwargs,
    ) -> Tuple[Union[RND_OE, RND_OI, RND_T, RND_TO, RND_TO_L], DictConfig]:
        """Initialize the RND model based on the model name and configuration.

        Args:
            model_name (str): The name of the model to initialize.
            cfg (DictConfig): The configuration for the model.

        Returns:
            Tuple[Union[RND_OE, RND_OI, RND_T, RND_TO, RND_TO_L], DictConfig]: The initialized model and its configuration.
        """
        class_name = model_name.upper()
        module_name = "rnd." + class_name
        state_dim = self.dataset.get_tensor_shape("states")[-1]
        obs_embedding_dim = self.dataset.get_tensor_shape("obs_embeddings")[-1]
        if is_pooled(cfg) and "obs_embeddings" in list(cfg.required_tensors):
            # Pooled (and possibly aligned) embeddings may be wider than this task's own.
            obs_embedding_dim = int(self._pooled_tensors(model_name, cfg)[0]["obs_embeddings"].shape[-1])
        elif is_demo_pooled(cfg) and self._demo_aligner(cfg) is not None:
            obs_embedding_dim = self._demo_aligner(cfg).target_dim
        action_pred_shape = self.dataset.get_tensor_shape(
            "action_preds",
            filter_actions=True,
            required_actions=cfg.required_actions,
            optional_actions=cfg.optional_actions,
        )[-3:]
        rgb_image_shape = self.dataset.get_tensor_shape("rgb_images")[1:]

        model_cfg = {
            "input_dict": {
                "input_size": None,
                "output_size": 512,
                "hyperparameters": cfg.model_hyperparameters,
                "use_states": state_dim > 0,
                # Used for calculating input sizes
                "state_dim": state_dim,
                "obs_embedding_dim": obs_embedding_dim,
                "action_pred_shape": action_pred_shape,
                "rgb_image_shape": rgb_image_shape,
                # Determines the action prediction handling
                "action_batch_handling": cfg.action_batch_handling,
                "action_execution_horizon": self.task_cfg.task.action_space.action_execution_horizon,
                "normalize_tensors": cfg.normalize_tensors,
            },
            # Used to identify the model
            "hparams": cfg.hparams.model,
            # Used to load the model
            "model_type": class_name,
            "_target_": module_name,
            "cfg": cfg,
            "kwargs": kwargs,
        }

        rnd_class = hydra.utils.get_class(module_name)

        model: RNDBase = rnd_class(input_dict=model_cfg["input_dict"], **kwargs).to(self.device)
        # Model calculates some parameters based on the input_dict
        model_cfg["input_dict"] = model.input_dict
        return model, model_cfg

    def _training_loop(
        self,
        save_dir: str,
        cfg: DictConfig,
        model_name: str,
        **kwargs,
    ):
        """Modularized function that:
            - Loads a model and model_configuration
            - Loads the model-specific datasets
            - Initializes the optimizer
            - Initializes the training parameters
            - Initializes the dataloaders (train and optimally validation)
            - Trains the model
            - Saves the model checkpoints
        Args:
            save_dir (str): The directory to save the model checkpoints.
            cfg (DictConfig): The configuration for the model.
        """
        # Load the model
        model, model_cfg = self._init_rnd_model(model_name, cfg, **kwargs)
        model_exists = model._check_for_existance(save_dir, model_cfg["hparams"])
        if model_exists:
            print(f"RND Model {model_name} already exists. Skipping training.")
            return
        dataset_dict = self._get_datasets_for_model(model_cfg["model_type"], cfg, **kwargs)
        optimizer, scheduler = self._get_optimizers(model, cfg)

        # Initialize training parameters
        best_loss = float("inf")
        patience = cfg.rnd_train.get("patience", 5)  # Default patience for early stopping
        keep_checkpoints = cfg.rnd_train.get("keep_checkpoints", 2)
        checkpoint_counter = 0
        patience_counter = 0
        save_every_n_epochs = cfg.rnd_train.get("save_every_n_epochs", 0)  # 0 Means no saving
        early_stopping = cfg.rnd_train.get("early_stopping", True)
        use_validation = cfg.rnd_train.get("use_validation", False)

        if use_validation:
            train_set, val_set = self._split_datasets(dataset_dict, train_ratio=cfg.rnd_train.get("train_ratio", 0.9))
        else:
            train_set = dataset_dict

        # Create DataLoader
        print("Datasets used for training:")
        for key in dataset_dict.keys():
            print(f"{key} with shape {dataset_dict[key].shape}")

        train_dataloader = self._create_dataloader(
            train_set,
            batch_size=min(cfg.rnd_train.batch_size, train_set[list(dataset_dict.keys())[0]].shape[0]),
            shuffle=True,
        )

        if use_validation:
            val_dataloader = self._create_dataloader(
                val_set,
                batch_size=min(cfg.rnd_train.batch_size, val_set[list(dataset_dict.keys())[0]].shape[0]),
                shuffle=False,
            )

        if early_stopping:
            print(f"Early stopping with patience {patience} and saving every {save_every_n_epochs} epochs.")
        else:
            print(
                f"No early stopping. Saving every {save_every_n_epochs} epochs (Zero means never save intermediate checkpoints)."
            )

        # Training loop
        train_losses, val_losses = [], []
        progress_bar = tqdm(range(cfg.rnd_train.n_epochs), desc="Training Progress", leave=False)

        best_state_dict = None

        for epoch in progress_bar:
            model.train()
            epoch_loss = 0.0
            for batch in train_dataloader:
                batch_dict = {key: value.to(self.device) for key, value in zip(dataset_dict.keys(), batch)}
                loss = model(**model.datasets_to_model_inputs(batch_dict)).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            if torch.isnan(loss).any():
                print(f"NaN loss detected at epoch {epoch + 1}.")
                return

            # Compute average training loss
            epoch_loss /= len(train_dataloader)
            train_losses.append(epoch_loss)

            # Validation
            if use_validation:
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for batch in val_dataloader:
                        batch_dict = {key: value.to(self.device) for key, value in zip(dataset_dict.keys(), batch)}
                        loss = model(**model.datasets_to_model_inputs(batch_dict)).mean()
                        val_loss += loss.item()
                    val_loss /= len(val_dataloader)
                    # val_loss = model(*val_set).mean().item()

                progress_bar.set_description(
                    f"Epoch {epoch + 1}: Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
                # print(f"Epoch {epoch + 1}: Train Loss = {epoch_loss:.4f}, Val Loss = {val_loss:.4f}")
            else:
                val_loss = epoch_loss
                progress_bar.set_description(f"Epoch {epoch + 1}: Train Loss: {epoch_loss:.4f}")
                # print(f"Epoch {epoch + 1}: Train Loss = {epoch_loss:.4f}")
            val_losses.append(val_loss)

            # Early stopping
            if early_stopping:
                if cfg.rnd_train.get("stop_when_avg_improvement", 0) > 0 and epoch > 15:
                    improvement = val_losses[-patience] - val_losses[-1]
                    if improvement < cfg.rnd_train.stop_when_avg_improvement:
                        print(f"Early stopping triggered due to improvement over patience being {improvement}.")
                        break
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    best_state_dict = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(
                            f"Early stopping triggered due to validation loss not improving over the last {patience_counter} epochs."
                        )
                        break
                if cfg.rnd_train.get("stop_when_val_to_train_ratio", 0) > 0 and epoch > 20:
                    val_to_train_ratio = val_loss / epoch_loss
                    if val_to_train_ratio > cfg.rnd_train.stop_when_val_to_train_ratio:
                        print(
                            f"Early stopping triggered due to validation to training loss ratio being greater than {cfg.rnd_train.stop_when_val_to_train_ratio}."
                        )
                        break
            # Save model checkpoint
            if save_every_n_epochs > 0:
                if (epoch + 1) % save_every_n_epochs == 0:
                    model._save_checkpoint(
                        save_dir,
                        model_cfg=model_cfg,
                        checkpoint_name=f"ckpt_epoch_{epoch + 1}.ckpt",
                        state_dict=best_state_dict,
                        kwargs=kwargs,
                    )
                    checkpoint_counter += 1
                    if checkpoint_counter > keep_checkpoints:
                        print("Removing old checkpoints")
                        os.remove(
                            os.path.join(
                                save_dir, f"ckpt_epoch_{epoch + 1 - keep_checkpoints * save_every_n_epochs}.ckpt"
                            )
                        )
                        checkpoint_counter -= 1
        # Debug print for rnd_twm model
        if hasattr(model, "last_error_influences"):
            print(
                "last_error_influences:",
                model.last_error_influences,
                "desired_influences:",
                model.target_error_influences,
                "last_norm_factors:",
                model.last_norm_factors,
            )

        if not early_stopping:
            print("Training finished without early stopping. Saving the final model.")
            best_state_dict = copy.deepcopy(model.state_dict())
        # Save the best model
        filename = f"best_{datetime.now().strftime('%Y-%m-%d_%H')}_epoch_{epoch + 1}_loss_{val_loss:.2g}.ckpt"

        model._save_checkpoint(
            save_dir,
            model_cfg=model_cfg,
            checkpoint_name=filename,
            state_dict=best_state_dict,
            kwargs=kwargs,
        )
        # # Plot training and validation loss
        self._plot_training_progress(train_losses, val_losses, save_dir)

    def _set_seed(self, seed):
        """Set the seed for reproducibility."""
        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def _save_tensors(self, save_dir: str, tensor, filename: str, overwrite: bool = True):
        """Save the tensor to a file."""
        os.makedirs(save_dir, exist_ok=True)
        if not filename.endswith(".pt"):
            filename += ".pt"
        if os.path.exists(os.path.join(save_dir, filename)) and not overwrite:
            return

        with open(os.path.join(save_dir, filename), "wb") as f:
            torch.save(tensor, f)

    def _load_tensors(self, load_dir, filename):
        if not os.path.exists(load_dir):
            raise ValueError(f"Data directory {load_dir} not found")
        if not os.path.exists(os.path.join(load_dir, filename)):
            raise ValueError(f"File {filename} not found in {load_dir}. Files in directory: {os.listdir(load_dir)}")
        with open(os.path.join(load_dir, filename), "rb") as f:
            tensor = torch.load(f, weights_only=True)
        return tensor

    def _plot_training_progress(self, train_losses, val_losses, save_dir):
        """Plot training and validation loss."""

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Progress")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_dir, "training_progress.png"))
        plt.close()

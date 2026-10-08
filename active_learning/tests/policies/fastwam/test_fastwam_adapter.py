"""FastWAM uncertainty adapter: the action branch denoised against a fixed video K/V cache.

Uses tiny randomly initialised DiTs and a fake VAE (no weights, no text encoder), so the tests
exercise the real MoT prefill / cached action-denoise path and the scheduler convention only.
"""

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("transformers", reason="fastwam requires the `fastwam` extra (transformers)")
pytest.importorskip("diffusers", reason="fastwam requires the `fastwam` extra (diffusers)")

from lerobot.policies.common.flow_matching.ode_solver import ODESolver
from lerobot.policies.factory import make_flow_matching_adapter, make_flow_matching_adapter_from_policy
from lerobot.policies.fastwam.configuration_fastwam import (
    FastWAMConfig,
    default_action_dit_config,
    default_video_dit_config,
)
from lerobot.policies.fastwam.fastwam_adapter import FastWAMAdapter
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
from lerobot.policies.fastwam.wan import ActionDiT, FastWAM, MoT, WanVideoDiT
from lerobot.utils.constants import OBS_STATE

ACTION_DIM, PROPRIO_DIM, TEXT_DIM, HORIZON, IMG = 7, 8, 16, 16, 64
LATENT_CH = 48


class FakeVAE(nn.Module):
    """Deterministic stand-in for the Wan VAE: 16x spatial downsample to 48 latent channels."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.register_buffer("proj", torch.randn(LATENT_CH, 3, generator=g) / 3)

    def encode(self, videos, device=None, tiled=False, tile_size=None, tile_stride=None):
        if isinstance(videos, (list, tuple)):
            videos = torch.stack(list(videos))  # [B, 3, T, H, W]
        pooled = torch.nn.functional.avg_pool3d(videos.to(self.proj), kernel_size=(1, 16, 16))
        return torch.einsum("bcthw,lc->blthw", pooled, self.proj)


def _tiny_config() -> FastWAMConfig:
    vid, act = default_video_dit_config(ACTION_DIM), default_action_dit_config(ACTION_DIM)
    for c in (vid, act):
        c.update(hidden_dim=64, ffn_dim=128, num_heads=2, attn_head_dim=32, num_layers=2, text_dim=TEXT_DIM)
    return FastWAMConfig(
        action_dim=ACTION_DIM,
        proprio_dim=PROPRIO_DIM,
        action_horizon=HORIZON,
        n_action_steps=8,
        num_video_frames=17,
        image_size=(IMG, IMG),
        load_text_encoder=False,
        torch_dtype="float32",
        device="cpu",
        video_dit_config=vid,
        action_dit_config=act,
        num_inference_steps=4,
    )


def _tiny_core(config: FastWAMConfig) -> FastWAM:
    torch.manual_seed(0)
    video_expert = WanVideoDiT(**config.video_dit_config)
    action_expert = ActionDiT(**config.action_dit_config)
    mot = MoT(mixtures={"video": video_expert, "action": action_expert}, mot_checkpoint_mixed_attn=False)
    return FastWAM(
        video_expert=video_expert,
        action_expert=action_expert,
        mot=mot,
        vae=FakeVAE(),
        text_encoder=None,
        tokenizer=None,
        text_dim=TEXT_DIM,
        proprio_dim=PROPRIO_DIM,
        device="cpu",
        torch_dtype=torch.float32,
    )


def _observation(batch_size: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    return {
        "observation.images.cam": torch.rand(batch_size, 3, IMG, IMG),
        OBS_STATE: torch.randn(batch_size, PROPRIO_DIM),
        "context": torch.randn(batch_size, 5, TEXT_DIM),
        "context_mask": torch.ones(batch_size, 5, dtype=torch.bool),
    }


@pytest.fixture(scope="module")
def adapter() -> FastWAMAdapter:
    config = _tiny_config()
    core = _tiny_core(config).eval()
    return make_flow_matching_adapter(model=core, policy_config=config)


def test_factory_dispatch(adapter, monkeypatch):
    assert isinstance(adapter, FastWAMAdapter)
    config = _tiny_config()
    monkeypatch.setattr(FastWAMPolicy, "_build_core_model", lambda self, cfg: _tiny_core(cfg))
    policy = FastWAMPolicy(config)
    from_policy = make_flow_matching_adapter_from_policy(policy)
    assert isinstance(from_policy, FastWAMAdapter)
    assert from_policy.model is policy.model
    assert from_policy.horizon == HORIZON
    assert from_policy.action_dim == ACTION_DIM
    assert from_policy.env_action_dim == ACTION_DIM
    assert from_policy.dtype == torch.float32


def test_conditioning_shapes_and_sample_expansion(adapter):
    batch_size, num_samples = 2, 3
    cond = adapter.prepare_conditioning(_observation(batch_size), num_samples)

    n_layers = adapter.config.action_dit_config["num_layers"]
    assert len(cond["video_kv_cache"]) == n_layers
    for layer in cond["video_kv_cache"]:
        assert layer["k"].shape[0] == batch_size * num_samples
        assert layer["k"].shape[1] == cond["video_seq_len"]
    # 64x64 image -> 4x4 latent -> 2x2 patches -> 4 video tokens, all first-frame
    assert cond["video_seq_len"] == 4
    assert cond["context"].shape == (batch_size * num_samples, 5 + 1, TEXT_DIM)  # + proprio token
    assert cond["attention_mask"].shape == (4 + HORIZON, 4 + HORIZON)
    assert cond["first_frame_latents"].shape == (batch_size, LATENT_CH, 1, IMG // 16, IMG // 16)

    # Row b*N+n of the expanded conditioning must be observation b: compare against single-observation
    # conditioning for the same noise.
    v_all = adapter.make_velocity_fn(cond)
    x = adapter.sample_prior(batch_size * num_samples, generator=torch.Generator().manual_seed(5))
    t = torch.tensor(0.4)
    out_all = v_all(t, x)
    assert out_all.shape == (batch_size * num_samples, HORIZON, ACTION_DIM)
    obs = _observation(batch_size)
    for b in range(batch_size):
        single = {k: v[b : b + 1] for k, v in obs.items()}
        v_b = adapter.make_velocity_fn(adapter.prepare_conditioning(single, 1))
        for n in range(num_samples):
            row = b * num_samples + n
            torch.testing.assert_close(out_all[row], v_b(t, x[row : row + 1])[0], atol=1e-5, rtol=1e-5)


def test_velocity_reproduces_infer_action(adapter):
    """Euler on FastWAM's own shifted sigma grid with the adapter's velocity == `infer_action`."""
    core, seed, steps = adapter.model, 3, 4
    obs = _observation(1)
    cond = adapter.prepare_conditioning(obs, 1)
    v_t = adapter.make_velocity_fn(cond)

    x = core._make_action_latents(HORIZON, seed, "cpu")
    timesteps, deltas = core.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=steps, device=core.device, dtype=x.dtype
    )
    n_train = float(core.infer_action_scheduler.num_train_timesteps)
    for step_t, delta in zip(timesteps, deltas, strict=True):
        t = 1.0 - step_t / n_train  # scheduler sigma -> shared time
        # scheduler: x += pred * delta (delta < 0 along sigma); v = -pred, so x += v * (-delta)
        x = x + v_t(t, x) * (-delta)

    ref = core.infer_action(
        prompt=None,
        input_image=obs["observation.images.cam"],
        action_horizon=HORIZON,
        proprio=obs[OBS_STATE],
        context=obs["context"],
        context_mask=obs["context_mask"],
        num_inference_steps=steps,
        seed=seed,
    )["action"]
    torch.testing.assert_close(x[0], ref, atol=1e-5, rtol=1e-5)
    assert ref.abs().sum() > 0


def test_ode_solver_integration_and_fiper_embedding(adapter):
    batch_size, num_samples = 2, 2
    cond = adapter.prepare_conditioning(_observation(batch_size), num_samples)
    solver_cfg = adapter.ode_solver_config
    x_0 = adapter.sample_prior(batch_size * num_samples, generator=torch.Generator().manual_seed(0))
    x_1 = ODESolver().sample(
        x_0=x_0,
        velocity_fn=adapter.make_velocity_fn(cond),
        method=solver_cfg["solver_method"],
        atol=solver_cfg["atol"],
        rtol=solver_cfg["rtol"],
        step_size=solver_cfg["step_size"],
    )
    assert x_1.shape == (batch_size * num_samples, HORIZON, ACTION_DIM)
    assert torch.isfinite(x_1).all()

    emb0 = adapter.prepare_fiper_obs_embedding(cond, batch_index=0)
    emb1 = adapter.prepare_fiper_obs_embedding(cond, batch_index=1)
    assert isinstance(emb0, np.ndarray) and emb0.dtype == np.float32 and emb0.ndim == 1
    assert emb0.shape == emb1.shape
    assert not np.allclose(emb0, emb1)  # different observations -> different embeddings


def test_prompt_via_precomputed_table(adapter):
    core = adapter.model
    obs = _observation(2)
    ctx, mask = obs.pop("context"), obs.pop("context_mask")
    obs["task"] = ["pick up the mug", "close the drawer"]
    prompts = [adapter.config.prompt_template.format(task=t) for t in obs["task"]]
    with pytest.raises(ValueError, match="text encoder"):
        adapter.prepare_conditioning(obs, 1)
    core.set_prompt_context_table(prompts, ctx, mask)
    try:
        via_table = adapter.prepare_conditioning(obs, 1)
        via_context = adapter.prepare_conditioning({**obs, "context": ctx, "context_mask": mask}, 1)
        torch.testing.assert_close(via_table["context"], via_context["context"])
        obs["task"] = ["pick up the mug", "an unseen instruction"]
        with pytest.raises(ValueError, match="not in the precomputed"):
            adapter.prepare_conditioning(obs, 1)
    finally:
        del core._prompt_context_index, core._prompt_context, core._prompt_context_mask


def test_frozen_video_expert_checkpoint_is_trainable_only_and_reloads_from_base(tmp_path, monkeypatch):
    """With freeze_video_expert the checkpoint omits the frozen tensors; from_pretrained refills them
    from base_model_id, giving back the exact state dict."""
    from safetensors import safe_open

    monkeypatch.setattr(FastWAMPolicy, "_build_core_model", lambda self, cfg: _tiny_core(cfg))
    base_dir = tmp_path / "base"
    full = FastWAMPolicy(_tiny_config())  # full save = the "base"
    full.save_pretrained(base_dir)
    with safe_open(base_dir / "model.safetensors", framework="pt") as f:
        n_full = len(list(f.keys()))

    cfg = _tiny_config()
    cfg.freeze_video_expert = True
    cfg.base_model_id = str(base_dir)
    cfg.pretrained_path = None
    tuned = FastWAMPolicy(cfg)
    tuned.model.load_state_dict(full.model.state_dict())
    # perturb the trainable part so the round-trip is a real test
    with torch.no_grad():
        for p in tuned.get_optim_params():
            p.add_(0.01)
    tuned_dir = tmp_path / "tuned"
    tuned.save_pretrained(tuned_dir)
    with safe_open(tuned_dir / "model.safetensors", framework="pt") as f:
        n_saved = len(list(f.keys()))
        assert f.metadata()["fastwam_frozen_from"] == str(base_dir)
    n_frozen = sum(1 for _, p in tuned.named_parameters() if not p.requires_grad)
    assert n_frozen > 0 and n_saved == n_full - n_frozen
    assert (tuned_dir / "model.safetensors").stat().st_size < 0.6 * (base_dir / "model.safetensors").stat().st_size

    reloaded = FastWAMPolicy.from_pretrained(tuned_dir)
    for k, v in tuned.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[k], v, atol=0, rtol=0)


def test_predict_action_chunk_batched_matches_per_sample(monkeypatch):
    """Batched inference (one prefill, batched denoising) equals `infer_action` run per sample."""
    config = _tiny_config()
    monkeypatch.setattr(FastWAMPolicy, "_build_core_model", lambda self, cfg: _tiny_core(cfg))
    policy = FastWAMPolicy(config).eval()
    obs = _observation(3)
    batched = policy.predict_action_chunk(obs)
    assert batched.shape == (3, HORIZON, ACTION_DIM)
    for b in range(3):
        ref = policy.model.infer_action(
            prompt=None,
            input_image=obs["observation.images.cam"][b : b + 1],
            action_horizon=HORIZON,
            proprio=obs[OBS_STATE][b : b + 1],
            context=obs["context"][b : b + 1],
            context_mask=obs["context_mask"][b : b + 1],
            num_inference_steps=config.num_inference_steps,
            seed=config.inference_seed,
        )["action"]
        torch.testing.assert_close(batched[b], ref, atol=1e-5, rtol=1e-5)


def test_frozen_checkpoint_refills_from_legacy_named_base(tmp_path, monkeypatch):
    """Released bases key blocks as `mot.mixtures.<expert>.blocks.<i>.*`; the refill must map them."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    from lerobot.policies.fastwam.modeling_fastwam import _legacy_block_key

    assert _legacy_block_key("model.mot.layers.3.blocks.video.ffn.0.weight") == "model.mot.mixtures.video.blocks.3.ffn.0.weight"
    assert _legacy_block_key("model.mot.mixtures.video.head.weight") is None

    monkeypatch.setattr(FastWAMPolicy, "_build_core_model", lambda self, cfg: _tiny_core(cfg))
    full = FastWAMPolicy(_tiny_config())
    full.save_pretrained(tmp_path / "modern")
    state = load_file(tmp_path / "modern" / "model.safetensors")
    legacy_state = {(_legacy_block_key(k) or k): v for k, v in state.items()}
    assert len(legacy_state) == len(state) and any(".mixtures.video.blocks." in k for k in legacy_state)
    base_dir = tmp_path / "legacy_base"
    base_dir.mkdir()
    save_file(legacy_state, base_dir / "model.safetensors")
    (tmp_path / "modern" / "config.json").rename(base_dir / "config.json")

    cfg = _tiny_config()
    cfg.freeze_video_expert = True
    cfg.base_model_id = str(base_dir)
    cfg.pretrained_path = None
    tuned = FastWAMPolicy(cfg)
    tuned.model.load_state_dict(full.model.state_dict())
    tuned.save_pretrained(tmp_path / "tuned")
    reloaded = FastWAMPolicy.from_pretrained(tmp_path / "tuned")
    for k, v in tuned.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[k], v, atol=0, rtol=0)


def test_frozen_tensors_are_shared_between_policies_loaded_from_one_base(tmp_path, monkeypatch):
    monkeypatch.setattr(FastWAMPolicy, "_build_core_model", lambda self, cfg: _tiny_core(cfg))
    base_dir = tmp_path / "base"
    FastWAMPolicy(_tiny_config()).save_pretrained(base_dir)
    cfg = _tiny_config()
    cfg.freeze_video_expert = True
    cfg.base_model_id = str(base_dir)
    cfg.pretrained_path = None
    tuned = FastWAMPolicy(cfg)
    tuned.save_pretrained(tmp_path / "tuned")
    a = FastWAMPolicy.from_pretrained(tmp_path / "tuned")
    b = FastWAMPolicy.from_pretrained(tmp_path / "tuned")
    frozen = [n for n, p in a.named_parameters() if not p.requires_grad]
    trainable = [n for n, p in a.named_parameters() if p.requires_grad]
    assert frozen and trainable
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert all(pa[n].data_ptr() == pb[n].data_ptr() for n in frozen), "frozen tensors must share storage"
    assert all(pa[n].data_ptr() != pb[n].data_ptr() for n in trainable), "trainable tensors must not"
    for n, v in tuned.state_dict().items():
        torch.testing.assert_close(pa[n] if n in pa else a.state_dict()[n], v, atol=0, rtol=0)


def test_clear_frozen_base_cache_releases_the_shared_tensors(tmp_path, monkeypatch):
    import gc
    import weakref

    from lerobot.policies.fastwam import modeling_fastwam as mf

    monkeypatch.setattr(FastWAMPolicy, "_build_core_model", lambda self, cfg: _tiny_core(cfg))
    base_dir = tmp_path / "base"
    FastWAMPolicy(_tiny_config()).save_pretrained(base_dir)
    cfg = _tiny_config()
    cfg.freeze_video_expert = True
    cfg.base_model_id = str(base_dir)
    cfg.pretrained_path = None
    FastWAMPolicy(cfg).save_pretrained(tmp_path / "tuned")

    mf.clear_frozen_base_cache()
    policy = FastWAMPolicy.from_pretrained(tmp_path / "tuned")
    assert mf._FROZEN_BASE_CACHE, "loading a trainable-only checkpoint must populate the shared cache"
    cached = next(iter(next(iter(mf._FROZEN_BASE_CACHE.values())).values()))
    ref = weakref.ref(cached)
    del policy, cached
    gc.collect()
    assert ref() is not None, "the cache keeps the frozen tensors alive after the policy is freed"
    mf.clear_frozen_base_cache()
    gc.collect()
    assert ref() is None, "clearing the cache must release them"

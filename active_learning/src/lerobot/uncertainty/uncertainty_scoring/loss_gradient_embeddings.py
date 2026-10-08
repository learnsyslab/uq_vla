"""Last-layer loss-gradient embeddings.

For a policy whose final layer maps features ``h`` to a prediction ``W h``, the gradient of the
per-sample squared loss with respect to ``W`` is ``2 (prediction - target) h^T``. Flattening that
outer product yields an embedding whose inner products define the linearized (NTK) kernel used by
AMF-style posterior-covariance selection, see https://arxiv.org/abs/2410.05026.

The embedding is obtained from a single forward pass: no backward pass is required because the
last-layer gradient is available in closed form.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_STATE

ACTION_IS_PAD = "action_is_pad"


def compute_loss_gradient_embeddings(
    policy,
    batch: dict[str, Tensor],
    *,
    seed: int,
    scope: str = "action_out_proj",
) -> Tensor:
    """Return one flattened last-layer loss gradient per batch element, shaped ``(batch, dim)``.

    Args:
        policy: A policy whose type is registered in ``_EXTRACTORS``.
        batch: A preprocessed training batch, as consumed by ``policy.forward``.
        seed: Seed for the flow-matching ``(time, noise)`` draw, so embeddings are reproducible.
        scope: Which last layer to differentiate.
    """
    extractor = _EXTRACTORS.get(policy.name)
    if extractor is None:
        raise NotImplementedError(
            f"Loss-gradient embeddings are not implemented for policy type {policy.name!r}. "
            f"Supported types: {sorted(_EXTRACTORS)}."
        )
    return extractor(policy, batch, seed=seed, scope=scope)


def is_supported_policy_type(policy_type: str) -> bool:
    return policy_type in _EXTRACTORS


def _seeded_flow_inputs(model, actions: Tensor, seed: int) -> tuple[Tensor, Tensor]:
    """Draw ``(noise, time)`` from the model's own samplers without disturbing global RNG state."""
    device = actions.device
    # `manual_seed` touches every CUDA device, so fork all of them to leave global RNG untouched.
    devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        noise = model.sample_noise(actions.shape, device)
        time = model.sample_time(actions.shape[0], device)
    return noise, time


def _smolvla_embeddings(policy, batch: dict[str, Tensor], *, seed: int, scope: str) -> Tensor:
    if scope != "action_out_proj":
        raise NotImplementedError(
            f"SmolVLA loss-gradient embeddings only support scope='action_out_proj', got {scope!r}."
        )

    model = policy.model
    if policy.config.adapt_to_pi_aloha:
        from lerobot.policies.common.aloha import pi_aloha_decode_state, pi_aloha_encode_actions_inv

        batch = dict(batch)
        batch[OBS_STATE] = pi_aloha_decode_state(batch[OBS_STATE])
        batch[ACTION] = pi_aloha_encode_actions_inv(batch[ACTION])

    actions = model.prepare_action(batch)
    noise, time = _seeded_flow_inputs(model, actions, seed)

    captured: dict[str, Tensor] = {}

    def _capture(_module, inputs, output):
        captured["features"] = inputs[0].detach()
        captured["prediction"] = output.detach()

    handle = model.action_out_proj.register_forward_hook(_capture)
    try:
        with torch.no_grad():
            model.forward(batch, noise=noise, time=time)
    finally:
        handle.remove()

    # Target velocity of the optimal-transport path, matching VLAFlowMatching.forward.
    residual = captured["prediction"] - (noise - actions)
    residual = residual[..., : policy.config.action_feature.shape[0]]
    actions_is_pad = batch.get(ACTION_IS_PAD)
    if actions_is_pad is not None:
        residual = residual * (~actions_is_pad).unsqueeze(-1)

    # d/dW of mean_chunk ||W h - u||^2 is 2 * mean_chunk (v - u) h^T.
    features = captured["features"]
    chunk_size = residual.shape[1]
    outer = torch.einsum("bca,bch->bah", residual, features) * (2.0 / chunk_size)
    return outer.flatten(start_dim=1)


def _xvla_embeddings(policy, batch: dict[str, Tensor], *, seed: int, scope: str) -> Tensor:
    if scope != "action_out_proj":
        raise NotImplementedError(
            f"X-VLA loss-gradient embeddings only support scope='action_out_proj', got {scope!r}."
        )

    model = policy.model
    # X-VLA's last layer is `transformer.action_decoder`, a DomainAwareLinear: per sample it is
    # an ordinary affine map `pred = W_d h + b_d` whose weights are looked up by domain id. The
    # outer-product form of the last-layer gradient therefore still holds, with `W_d` as W.
    decoder = model.transformer.action_decoder

    captured: dict[str, Tensor] = {}

    def _capture(_module, inputs, output):
        captured["features"] = inputs[0].detach()
        captured["prediction"] = output.detach()

    inputs = policy._build_model_inputs(batch)
    targets = policy._prepare_action_targets(batch)

    handle = decoder.register_forward_hook(_capture)
    try:
        # `XVLAModel.forward` draws its own flow time as `(rand(1) + arange(B)/B) % (1 - 1e-5)`,
        # a single shared offset stratified across the batch, so seeding the RNG around the call
        # is what makes these embeddings reproducible. Fork so global RNG is left untouched.
        device = targets.device
        devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            model.forward(action=targets, **inputs)
    finally:
        handle.remove()

    # Unlike SmolVLA's plain squared loss, X-VLA's objective is a weighted mix of MSE terms and a
    # BCE on the gripper logits, with per-space index groups and scales (see action_hub.py). Its
    # d(loss)/d(prediction) is therefore neither `2 (pred - target)` nor uniform across channels.
    # Rather than re-deriving it here -- which would silently go wrong whenever the configured
    # action space changes -- differentiate the model's own loss. The forward above already ran
    # under no_grad, so this builds a graph over the loss head alone, not the VLM.
    prediction = captured["prediction"].detach().requires_grad_(True)
    target = targets.to(dtype=prediction.dtype)
    losses = model.action_space.compute_loss(prediction, target)
    grad = torch.autograd.grad(sum(losses.values()), prediction)[0].detach()

    # compute_loss reduces over the batch, so `grad` carries a 1/batch factor that a per-sample
    # embedding should not have. Rescaling keeps the kernel independent of the batching used to
    # compute it, which matters because candidates are embedded in chunks.
    grad = grad * prediction.shape[0]

    # DomainAwareLinear stores its weight as (input_size, output_size) and applies `x @ W`, the
    # transpose of nn.Linear's (out, in). Contract in that order so the embedding really is
    # vec(dL/dW) for this layer. (Inner products -- all AMF consumes -- are invariant to a
    # consistent coordinate permutation, so this is about matching the parameter layout, not
    # about changing the kernel.)
    features = captured["features"]
    outer = torch.einsum("bch,bca->bha", features, grad.to(features.dtype))
    return outer.flatten(start_dim=1)


def _fastwam_embeddings(policy, batch: dict[str, Tensor], *, seed: int, scope: str) -> Tensor:
    if scope != "action_out_proj":
        raise NotImplementedError(
            f"FastWAM loss-gradient embeddings only support scope='action_out_proj', got {scope!r}."
        )

    model = policy.model
    # FastWAM's action branch ends in `action_expert.head`, a plain nn.Linear from the expert's
    # hidden width to the action channels, so the closed-form last-layer gradient applies exactly
    # as it does for SmolVLA. The video expert has its own head and its own loss term; only the
    # action head is differentiated here, which is the analogue of SmolVLA's `action_out_proj`.
    head = model.action_expert.head

    captured: dict[str, Tensor] = {}

    def _capture(_module, inputs, output):
        captured["features"] = inputs[0].detach()
        captured["prediction"] = output.detach()

    # `training_loss` builds its own targets and does not return them, so wrap the action-loss
    # method to record the target, the flow timestep and the padding mask it was handed.
    original_action_loss = model._compute_training_action_loss

    def _recording_action_loss(*, inputs, pred_action, target_action, timestep_action):
        captured["target"] = target_action.detach()
        captured["timestep"] = timestep_action.detach()
        action_is_pad = inputs.get("action_is_pad")
        captured["action_is_pad"] = None if action_is_pad is None else action_is_pad.detach()
        return original_action_loss(
            inputs=inputs,
            pred_action=pred_action,
            target_action=target_action,
            timestep_action=timestep_action,
        )

    handle = head.register_forward_hook(_capture)
    model._compute_training_action_loss = _recording_action_loss
    try:
        # FastWAM draws its own (noise, timestep) inside `_sample_training_targets` rather than
        # accepting them as arguments, so seeding around the call is what makes these embeddings
        # reproducible. Fork so global RNG is left untouched.
        device = next(model.parameters()).device
        devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            policy.forward(batch)
    finally:
        handle.remove()
        model._compute_training_action_loss = original_action_loss

    # FastWAM runs in bfloat16 and the einsum below contracts over the whole action chunk, so
    # accumulate in float32 to keep that contraction clean. This does not remove the ~1% gap to
    # an autograd reference: that gap is the bfloat16 head forward itself, and the embedding is
    # meant to be the gradient of what the model actually computes, so bfloat16 there is correct.
    prediction = captured["prediction"].to(torch.float32)
    residual = prediction - captured["target"].to(torch.float32)
    num_channels = residual.shape[-1]

    # d(loss_b)/d(pred) for FastWAM's action loss, which is a per-token MSE averaged over the
    # action channels, then averaged over the valid chunk positions, then scaled by a per-sample
    # flow-timestep weight. The 1/batch from the final mean is deliberately omitted so the
    # embedding is per-sample and independent of how candidates were chunked.
    scale = torch.full_like(residual[..., :1], 2.0 / num_channels)
    action_is_pad = captured.get("action_is_pad")
    if action_is_pad is not None:
        valid = (~action_is_pad).to(dtype=residual.dtype, device=residual.device)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        scale = scale * (valid / valid_sum.unsqueeze(1)).unsqueeze(-1)
    else:
        scale = scale / residual.shape[1]
    weight = model.train_action_scheduler.training_weight(captured["timestep"])
    weight = weight.reshape(-1).to(dtype=residual.dtype, device=residual.device)
    grad = residual * scale * weight.view(-1, 1, 1)

    # nn.Linear stores weight as (out_features, in_features), so contract to (action, hidden).
    features = captured["features"].to(grad.dtype)
    outer = torch.einsum("bca,bch->bah", grad, features)
    return outer.flatten(start_dim=1)


_EXTRACTORS = {
    "smolvla": _smolvla_embeddings,
    "xvla": _xvla_embeddings,
    "fastwam": _fastwam_embeddings,
}

"""The last-layer loss-gradient embedding must equal the true gradient wrt that layer's weight.

AMF's kernel is built from inner products of these embeddings, so an embedding that is not the
real gradient silently produces a plausible-looking but wrong acquisition criterion. These tests
pin the identity down for X-VLA, where two details are easy to get wrong:

  * the objective is a weighted mix of MSE terms and a BCE on the gripper logits, so
    d(loss)/d(prediction) is neither `2 (pred - target)` nor uniform across action channels, and
    it differs between the `joint` and `ee6d` action spaces;
  * `DomainAwareLinear` stores its weight as (input_size, output_size) and applies `x @ W`, the
    transpose of `nn.Linear`, so the outer product has to be contracted in that order for the
    embedding to be vec(dL/dW).
"""

import pytest
import torch

from lerobot.policies.xvla.action_hub import build_action_space
from lerobot.policies.xvla.soft_transformer import DomainAwareLinear
from lerobot.uncertainty.uncertainty_scoring.loss_gradient_embeddings import (
    is_supported_policy_type,
)


def _embedding(features, grad, batch_size):
    """The contraction performed by `_xvla_embeddings`, given captured features and dL/dpred."""
    return torch.einsum("bch,bca->bha", features, grad * batch_size).flatten(1)


@pytest.mark.parametrize(("space_name", "action_dim"), [("joint", 14), ("ee6d", 20)])
def test_xvla_embedding_equals_true_last_layer_gradient(space_name, action_dim):
    torch.manual_seed(0)
    batch, chunk, hidden = 3, 4, 8
    make_space = lambda: build_action_space(space_name).double()  # noqa: E731

    decoder = DomainAwareLinear(hidden, action_dim, num_domains=2).double()
    features = torch.randn(batch, chunk, hidden, dtype=torch.double)
    domain_id = torch.tensor([0, 1, 0])
    target = torch.randn(batch, chunk, action_dim, dtype=torch.double)

    prediction = decoder(features, domain_id).detach().requires_grad_(True)
    losses = make_space().compute_loss(prediction, target)
    grad = torch.autograd.grad(sum(losses.values()), prediction)[0]
    embedding = _embedding(features, grad, batch)

    # Ground truth: differentiate each sample's own loss wrt its domain's weight matrix.
    expected = []
    for i in range(batch):
        weight = decoder.fc(domain_id[i : i + 1]).view(hidden, action_dim).detach().clone()
        weight.requires_grad_(True)
        bias = decoder.bias(domain_id[i : i + 1]).view(action_dim).detach()
        sample_pred = (features[i] @ weight + bias).unsqueeze(0)
        sample_loss = sum(make_space().compute_loss(sample_pred, target[i : i + 1]).values())
        expected.append(torch.autograd.grad(sample_loss, weight)[0].flatten())
    expected = torch.stack(expected)

    assert embedding.shape == (batch, hidden * action_dim)
    torch.testing.assert_close(embedding, expected, rtol=0, atol=1e-9)


def test_embedding_inner_products_are_layout_invariant():
    """AMF consumes only inner products, so a consistent coordinate permutation must not matter."""
    torch.manual_seed(0)
    batch, chunk, hidden, action_dim = 3, 4, 8, 14
    features = torch.randn(batch, chunk, hidden, dtype=torch.double)
    grad = torch.randn(batch, chunk, action_dim, dtype=torch.double)

    as_stored = torch.einsum("bch,bca->bha", features, grad).flatten(1)
    transposed = torch.einsum("bca,bch->bah", grad, features).flatten(1)

    torch.testing.assert_close(as_stored @ as_stored.T, transposed @ transposed.T)


def test_supported_policy_types():
    assert is_supported_policy_type("smolvla")
    assert is_supported_policy_type("xvla")
    assert not is_supported_policy_type("act")

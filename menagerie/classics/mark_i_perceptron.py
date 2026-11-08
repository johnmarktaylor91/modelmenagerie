"""Mark-I / cross-coupled Perceptron (1962), Frank Rosenblatt.

Paper: "Principles of Neurodynamics."
The Mark I's sizes: a 20 x 20 photocell retina (400 sensory units), 512 association
units and 8 response units. Each association unit has a fixed set of random
excitatory or inhibitory connections from the retina and fires when its net input
reaches its threshold. The cross-coupled system adds fixed random association-to-
association connections, relaxed for one step. The adaptive association-to-response
weights are the only learned part; here they hold a seeded untrained draw, and each
response unit outputs 1 when its weighted sum is non-negative, else 0.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

SEED = 1962
RETINA_SIDE = 20


class MarkIPerceptron(nn.Module):
    """Rosenblatt S-A-R perceptron with fixed random wiring and cross-coupled A units."""

    def __init__(self, n_sensory: int = RETINA_SIDE * RETINA_SIDE, n_assoc: int = 512,
                 n_response: int = 8, fan_in: int = 40, threshold: float = 0.0,
                 coupling: float = 0.5, seed: int = SEED) -> None:
        """Initialize the fixed S-to-A and A-to-A wiring and the adaptive A-to-R layer.

        Parameters
        ----------
        n_sensory
            Number of sensory units (photocells).
        n_assoc
            Number of association units.
        n_response
            Number of response units.
        fan_in
            Random sensory connections per association unit, each +1 or -1.
        threshold
            Association-unit firing threshold.
        coupling
            Scale of the random association-to-association weights.
        seed
            Seed for the fixed wiring and the untrained response weights.
        """
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        sensory_assoc = torch.zeros(n_sensory, n_assoc)
        for unit in range(n_assoc):
            sources = torch.randperm(n_sensory, generator=gen)[:fan_in]
            signs = torch.randint(0, 2, (fan_in,), generator=gen).float() * 2 - 1
            sensory_assoc[sources, unit] = signs
        lateral = torch.randn(n_assoc, n_assoc, generator=gen) * coupling
        lateral.fill_diagonal_(0.0)
        self.register_buffer("sensory_assoc", sensory_assoc)
        self.register_buffer("lateral", lateral)
        self.threshold = threshold
        self.response = nn.Linear(n_assoc, n_response)
        bound = n_assoc ** -0.5
        with torch.no_grad():
            self.response.weight.copy_(
                (torch.rand(n_response, n_assoc, generator=gen) * 2 - 1) * bound)
            self.response.bias.copy_((torch.rand(n_response, generator=gen) * 2 - 1) * bound)

    def forward(self, x: Tensor) -> Tensor:
        """Compute thresholded association units, one cross-coupled step and the responses.

        Parameters
        ----------
        x
            Retina activity of shape ``(batch, n_sensory)``.

        Returns
        -------
        Tensor
            Binary response activations of shape ``(batch, n_response)``.
        """
        sensory_input = x @ self.sensory_assoc
        association = (sensory_input >= self.threshold).to(x.dtype)
        coupled = sensory_input + association @ self.lateral
        association = (coupled >= self.threshold).to(x.dtype)
        response_sum = self.response(association)
        return (response_sum >= 0.0).to(x.dtype)


def build() -> nn.Module:
    """Build the Mark-I perceptron at the machine's sizes.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return MarkIPerceptron()


def example_input() -> Tensor:
    """Return four seeded binary 20 x 20 retina patterns, flattened.

    Returns
    -------
    Tensor
        Example input tensor of shape ``(4, 400)``.
    """
    gen = torch.Generator().manual_seed(SEED + 1)
    return (torch.rand(4, RETINA_SIDE * RETINA_SIDE, generator=gen) < 0.5).float()

"""McCulloch-Pitts threshold logic net (1943), Warren McCulloch and Walter Pitts.

Paper: "A logical calculus of the ideas immanent in nervous activity."
All-or-none neurons fire when the number of active excitatory synapses reaches a fixed
threshold, and any active inhibitory synapse absolutely prevents firing. A net without
circles of such neurons realizes propositional logic, one synaptic delay per layer. This net
wires its units as the paper's elementary propositions (conjunction, disjunction, a conjunction
with an inhibitory veto, a two-of-three majority) and composes them in a second layer; the
example input is the complete truth table of its six input neurons.
"""

from __future__ import annotations

import itertools

import torch
from torch import Tensor, nn

N_IN = 6

# Hidden units over inputs x0..x5: (excitatory inputs, inhibitory inputs, threshold).
HIDDEN = (
    ((0, 1), (), 2.0),  # h0 = x0 AND x1
    ((2, 3), (), 1.0),  # h1 = x2 OR x3
    ((4,), (5,), 1.0),  # h2 = x4 AND NOT x5
    ((1, 3, 5), (), 2.0),  # h3 = at least two of x1, x3, x5
)

# Output units over hidden units h0..h3, same layout.
OUTPUT = (
    ((0, 1), (), 1.0),  # o0 = h0 OR h1
    ((1, 2), (), 2.0),  # o1 = h1 AND h2
    ((2, 3), (0,), 1.0),  # o2 = (h2 OR h3) AND NOT h0
)


def _synapses(units: tuple, n_from: int) -> tuple[Tensor, Tensor, Tensor]:
    """Excitatory and inhibitory 0/1 connection matrices and thresholds for a unit table."""
    exc = torch.zeros(len(units), n_from)
    inh = torch.zeros(len(units), n_from)
    for row, (excitatory, inhibitory, _) in enumerate(units):
        exc[row, list(excitatory)] = 1.0
        inh[row, list(inhibitory)] = 1.0
    theta = torch.tensor([threshold for _, _, threshold in units])
    return exc, inh, theta


class McCullochPittsNet(nn.Module):
    """Fixed two-layer threshold-logic net with absolute inhibitory veto connections."""

    def __init__(self) -> None:
        """Wire the fixed hidden and output threshold units from the unit tables."""
        super().__init__()
        hidden_exc, hidden_inh, hidden_theta = _synapses(HIDDEN, N_IN)
        out_exc, out_inh, out_theta = _synapses(OUTPUT, len(HIDDEN))
        self.register_buffer("hidden_exc", hidden_exc)
        self.register_buffer("hidden_inh", hidden_inh)
        self.register_buffer("hidden_theta", hidden_theta)
        self.register_buffer("out_exc", out_exc)
        self.register_buffer("out_inh", out_inh)
        self.register_buffer("out_theta", out_theta)

    @staticmethod
    def _threshold_layer(x: Tensor, exc: Tensor, inh: Tensor, theta: Tensor) -> Tensor:
        """Apply one layer of McCulloch-Pitts neurons.

        Parameters
        ----------
        x
            Binary activations of shape ``(batch, features)``.
        exc
            Excitatory connections, shape ``(units, features)``.
        inh
            Inhibitory connections, shape ``(units, features)``.
        theta
            Threshold of each unit.

        Returns
        -------
        Tensor
            Binary activations of shape ``(batch, units)``.
        """
        excitation = x @ exc.T
        inhibited = (x @ inh.T) > 0.0
        active = (excitation >= theta) & (~inhibited)
        return active.to(x.dtype)

    def forward(self, x: Tensor) -> Tensor:
        """Compute the output neurons' firing for each input state.

        Parameters
        ----------
        x
            Binary input tensor of shape ``(batch, 6)``.

        Returns
        -------
        Tensor
            Binary output tensor of shape ``(batch, 3)``.
        """
        hidden = self._threshold_layer(x, self.hidden_exc, self.hidden_inh, self.hidden_theta)
        return self._threshold_layer(hidden, self.out_exc, self.out_inh, self.out_theta)


def build() -> nn.Module:
    """Build the fixed McCulloch-Pitts threshold-logic net (no learned or random weights).

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return McCullochPittsNet()


def example_input() -> Tensor:
    """Return every binary state of the six input neurons (the net's truth table).

    Returns
    -------
    Tensor
        Tensor of shape ``(64, 6)`` holding 0.0 and 1.0.
    """
    return torch.tensor(list(itertools.product((0.0, 1.0), repeat=N_IN)))

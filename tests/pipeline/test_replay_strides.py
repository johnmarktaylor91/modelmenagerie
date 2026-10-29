"""Shallow depthwise convs on a token view replay exactly (AGT-592).

CoaT's and DaViT's ``ConvPosEnc`` feed a 3x3 depthwise conv ``tokens.transpose(1, 2).view(B, C,
H, W)``. With ``B = 1`` that view has batch stride ``C``: ATen reads it as NCHW (the conv runs the
standard kernel), yet ``is_contiguous(memory_format=torch.channels_last)`` holds. TorchLens up to
2.35.x copied such a payload with ``clone(memory_format=torch.channels_last)``, which re-strides
the batch dim, so the isolated replay ran the channels-last kernel and missed the saved output by
an ulp or so; with nine terms that cancel, the miss is far outside the 512-ulp relative replay
bound. TorchLens 2.36.0 copies with ``preserve_format`` (its commit 2e368833a) and the replay is
bit exact. The 13 Slice A rows (coat_*, davit_*) failed for this reason alone.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
tl = pytest.importorskip("torchlens")
from packaging.version import Version  # noqa: E402
from torch import nn  # noqa: E402

from menagerie.pipeline import worker  # noqa: E402

FIXED_IN = Version("2.36.0")
C, HW = 96, 14


class TokenPosEnc(nn.Module):
    """The ConvPosEnc shape: tokens -> (B, C, H, W) view -> 3x3 depthwise conv, residual."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(C, C)
        self.dw = nn.Conv2d(C, C, 3, padding=1, groups=C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        b, _, c = x.shape
        feat = x.transpose(1, 2).view(b, c, HW, HW)
        return self.dw(feat) + feat


def _subject() -> worker.Subject:
    worker._wrap_torch()
    torch.manual_seed(0)
    model = TokenPosEnc().eval()
    return worker.Subject(model, (torch.randn(1, HW * HW, C),), {})


def test_the_view_is_nchw_to_aten_and_a_channels_last_copy_changes_the_kernel() -> None:
    torch.manual_seed(0)
    conv = nn.Conv2d(C, C, 3, padding=1, groups=C)
    feat = torch.randn(1, HW * HW, C).transpose(1, 2).view(1, C, HW, HW)
    assert feat.stride()[0] == C and feat.is_contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        live = conv(feat)
        assert live.is_contiguous()  # ATen chose the standard (NCHW) kernel
        assert torch.equal(conv(feat.clone()), live)  # preserve_format: same strides, same bits
        restrided = conv(feat.clone(memory_format=torch.channels_last))
    if torch.equal(restrided, live):
        pytest.skip("this CPU's channels-last and NCHW depthwise kernels agree bit for bit")
    # The channels-last kernel rounds differently: an ulp, far more relative to a cancelled sum.
    assert (restrided - live).abs().max() < 1e-5


@pytest.mark.xfail(Version(tl.__version__) < FIXED_IN, strict=False,
                   reason="TorchLens < 2.36.0 re-strides the conv input payload (AGT-592); "
                          "whether that misses depends on the CPU's kernels")
@pytest.mark.parametrize("threads", [1, 3])
def test_token_view_depthwise_conv_validates(threads: int) -> None:
    check = worker._validate(_subject(), threads)
    assert check.passed, check.msg

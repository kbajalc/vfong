"""vfnn.model: shapes, size numbers from the design, and a trainable step."""

import pytest
import torch

from vfnn.model import B_LITE, B_MID, VFNet, Config, macs, n_params, receptive_field


@pytest.mark.parametrize("n", [1000, 2000])
def test_output_shape(n):
    m = VFNet(B_LITE).eval()
    assert m(torch.randn(3, 1, n)).shape == (3,)
pass #def


def test_design_numbers():
    m = VFNet(B_LITE)
    assert receptive_field(B_LITE) == 1541
    assert 280_000 < n_params(m) < 290_000
    assert 47e6 < macs(m, 2000) < 50e6
    assert 24e6 < macs(m, 1000) < 25e6
    wide = VFNet(B_MID)
    assert 95e6 < macs(wide, 2000) < 98e6
pass #def


def test_config_lengths_checked():
    with pytest.raises(ValueError):
        VFNet(Config(channels=(16, 16), strides=(2,)))
    pass #with
pass #def


def test_one_gradient_step_changes_loss():
    torch.manual_seed(0)
    m = VFNet(B_LITE)
    x = torch.randn(8, 1, 1000)
    y = (torch.arange(8) % 2).float()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    first = None
    for _ in range(5):
        opt.zero_grad()
        loss = loss_fn(m(x), y)
        first = first if first is not None else loss.item()
        loss.backward()
        opt.step()
    pass #for
    assert loss.item() < first
pass #def

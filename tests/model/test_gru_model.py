import pytest

torch = pytest.importorskip("torch")


def test_forward_pass_shape():
    from model.gru_model import SailingGRU
    model = SailingGRU()
    B, T, F = 4, 24, 7
    seq = torch.randn(B, T, F)
    ctx = torch.randn(B, 12)
    out = model(seq, ctx)
    assert out.shape == (B, 1)


def test_output_in_range():
    from model.gru_model import SailingGRU
    model = SailingGRU()
    seq = torch.randn(16, 24, 7)
    ctx = torch.randn(16, 12)
    out = model(seq, ctx)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_single_sample():
    from model.gru_model import SailingGRU
    model = SailingGRU()
    out = model(torch.randn(1, 24, 7), torch.randn(1, 12))
    assert out.shape == (1, 1)


def test_loss_is_scalar():
    from model.gru_model import SailingGRU
    model = SailingGRU()
    seq = torch.randn(8, 24, 7)
    ctx = torch.randn(8, 12)
    labels = torch.randint(0, 2, (8,)).float()
    out = model(seq, ctx).squeeze(1)
    loss = torch.nn.BCELoss()(out, labels)
    assert loss.ndim == 0
    assert loss.item() > 0

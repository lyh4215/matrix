import torch

from src.models.sinkhorn import sinkhorn


def test_square_sinkhorn_is_doubly_stochastic() -> None:
    torch.manual_seed(1)
    result = sinkhorn(torch.randn(2, 5, 5), iterations=80)
    assert torch.allclose(result.sum(-1), torch.ones(2, 5), atol=1e-4)
    assert torch.allclose(result.sum(-2), torch.ones(2, 5), atol=1e-4)


def test_rectangular_and_masked_sinkhorn_has_compatible_marginals() -> None:
    scores = torch.randn(2, 4, 5)
    row_mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    result = sinkhorn(scores, row_mask=row_mask, iterations=100)
    assert torch.allclose(result[0, :3].sum(-1), torch.ones(3), atol=1e-4)
    assert torch.all(result[0, 3] == 0)
    assert torch.allclose(result[0].sum(0), torch.full((5,), 3 / 5), atol=2e-4)
    assert torch.allclose(result[1].sum(0), torch.full((5,), 4 / 5), atol=2e-4)


import torch

from src.models.sinkhorn import sinkhorn


def test_square_sinkhorn_is_doubly_stochastic() -> None:
    torch.manual_seed(1)
    result = sinkhorn(torch.randn(2, 19, 19), iterations=80)
    assert torch.allclose(result.sum(-1), torch.ones(2, 19), atol=1e-4)
    assert torch.allclose(result.sum(-2), torch.ones(2, 19), atol=1e-4)


def test_rectangular_and_masked_sinkhorn_has_compatible_marginals() -> None:
    scores = torch.zeros(2, 5, 19)
    scores[0, 0, 0] = 10
    scores[0, 1, 1] = 10
    scores[0, 2, 2] = 10
    scores[1, 0, 0] = 10
    scores[1, 1, 1] = 10
    scores[1, 2, 2] = 10
    scores[1, 3, 3] = 10
    scores[1, 4, 4] = 10
    row_mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )
    result = sinkhorn(scores, row_mask=row_mask, iterations=100)
    assert torch.allclose(result[0, :3].sum(-1), torch.ones(3), atol=1e-4)
    assert torch.all(result[0, 3:] == 0)
    assert float(result[0].sum(0).max()) <= 1.0001
    assert float(result[1].sum(0).max()) <= 1.0001
    assert float(result[0, :, :3].sum()) > float(result[0, :, 3:].sum())


def test_partial_sinkhorn_limits_competing_real_rows_without_using_every_column() -> None:
    scores = torch.zeros(1, 5, 19)
    scores[:, :, 0] = 12.0
    result = sinkhorn(scores, iterations=150)
    assert torch.allclose(result.sum(-1), torch.ones(1, 5), atol=1e-4)
    assert float(result[0, :, 0].sum()) <= 1.0001
    assert float(result[0, :, 0].max()) < 0.5

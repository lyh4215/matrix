import pytest
import torch

from src.models.sequence_region_zone_matcher import SequenceRegionZoneMatcher
from src.models.region_zone_matcher import observed_permutation_nll


def model():
    return SequenceRegionZoneMatcher(d_model=16, num_layers=2, dropout=0, sinkhorn_iterations=100).eval()


def test_shape_one_to_one_reindexing_and_padding():
    torch.manual_seed(42)
    m = model()
    seq = torch.arange(19).repeat(3)[None].repeat(2, 1)
    output = m(seq)
    assert output.assignment_probabilities.shape == (2, 19, 19)
    assert torch.allclose(output.assignment_probabilities.sum(-1), torch.ones(2, 19), atol=1e-4)
    assert torch.allclose(output.assignment_probabilities.sum(-2), torch.ones(2, 19), atol=1e-4)
    reindex = torch.randperm(19)
    changed = m(reindex[seq])
    assert torch.allclose(changed.scores[:, reindex], output.scores, atol=1e-6)
    assert torch.allclose(changed.assignment_probabilities[:, reindex], output.assignment_probabilities, atol=1e-6)
    padded = m(torch.nn.functional.pad(seq, (0, 7), value=-1))
    assert torch.allclose(padded.scores, output.scores, atol=1e-6)


def test_partial_coverage_gradients_reach_attention_and_equality_bias():
    torch.manual_seed(7)
    m = model()
    seq = torch.tensor([[0, 1, 0, 2, 3, 1, -1], [3, 2, 2, 0, 1, 0, 3]])
    output = m(seq)
    observed = torch.zeros(2, 19, dtype=torch.bool)
    observed[:, :4] = True
    targets = torch.arange(19)[None].repeat(2, 1)
    loss = observed_permutation_nll(output.assignment_probabilities, targets, observed)
    assert torch.isfinite(loss)
    loss.backward()
    for layer in m.layers:
        for parameter in (layer.same_region_bias, layer.attention.in_proj_weight):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
    assert (output.assignment_probabilities[:, 4:] == 0).all()
    assert (output.assignment_probabilities.sum(-2) <= 1.001).all()


def test_sequence_order_changes_region_context():
    m = model()
    a = torch.tensor([[0, 1, 0, 2, 1, 2, 0, 2]])
    b = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2]])
    assert not torch.allclose(m(a).region_representations, m(b).region_representations)
    with pytest.raises(ValueError):
        m(torch.tensor([[-1, -1]]))


def test_training_reporting_and_checkpoint_roundtrip(tmp_path):
    import json
    from src.data.korean_corpus import build_corpus_graphs
    from src.benchmark.region_zone_match_probe import load_region_zone_probe_config, run_region_zone_match_probe
    config = load_region_zone_probe_config('configs/korean_sequence_benchmark.yaml')
    config.output_dir = str(tmp_path)
    config.learned.epochs = 1
    config.learned.device = 'cpu'
    config.learned.d_model = 8
    config.learned.num_layers = 1
    config.learned.dropout = 0
    config.synthetic.sequence_lengths = (19,)
    docs = {str(i): [''.join(chr(0xAC00 + (i * 71 + j * 587) % 11172) for j in range(57))] for i in range(30)}
    data = build_corpus_graphs(docs, {'train': 2, 'validation': 1, 'test': 1}, length=19)
    result = run_region_zone_match_probe(config, graph_data=data)
    assert [r['matcher'] for r in result['results']] == ['learned', 'learned_sequence']
    raw = json.loads((tmp_path / 'raw_results.json').read_text())
    assert len(raw['learned_sequence_history']) == 1
    checkpoint = torch.load(tmp_path / 'learned_sequence/checkpoint.pt', weights_only=True)
    restored = SequenceRegionZoneMatcher(d_model=8, num_layers=1, dropout=0)
    restored.load_state_dict(checkpoint['model_state'])
    assert checkpoint['matcher'] == 'learned_sequence'
    assert torch.isfinite(restored(torch.arange(19)[None]).assignment_probabilities).all()

"""Run matched-data GNN versus sequence-attention comparison."""
import sys
from pathlib import Path
from src.benchmark.korean_corpus_benchmark import main

if __name__ == '__main__':
    if '--config' not in sys.argv:
        sys.argv.extend(['--config', str(Path(__file__).parent / 'configs/korean_sequence_benchmark.yaml')])
    main()

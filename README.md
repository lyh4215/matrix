# Hangul Cipher Zone Neural Decoder

숫자 암호의 절대 위치가 아니라 **암호 공간의 국소 관계**와 **문장 문맥**으로
초성 기준 한글 zone을 추론하는 PyTorch 연구용 코드베이스입니다.

## 설치와 빠른 실행

```bash
python -m pip install -e '.[dev]'
python -m src.training.train --config configs/default.yaml --smoke
pytest
```

기본 설정은 합성 cipher table을 `f` 단위로 train/validation/test에 분리합니다.
실데이터는 한 줄에 한 episode인 JSONL로 넣을 수 있습니다.

```json
{"table_id":"f-1","cipher_values":[3124,1081],"zone_labels":[0,2],"cipher_zone_ids":[7,3]}
```

```bash
python -m src.training.train --config configs/default.yaml \
  --data-jsonl data/episodes.jsonl --baseline relational_sinkhorn
```

## 비교 가능한 baseline

`--baseline` 값 하나로 아래 모델을 선택합니다.

- `standard`: digit projection + standard Transformer + token classifier
- `relational`: relational self-attention + token classifier
- `relational_pool`: relational attention + cipher-zone pooling + classifier
- `relational_match`: pooling + Hangul prototype relation matching
- `relational_sinkhorn`: matching + rectangular Sinkhorn

설정 파일에서 relation feature, positional encoding, locality gate/mask, pooling,
cross matching, Sinkhorn, random offset, zone relocation을 각각 끌 수 있습니다.

## 핵심 인터페이스

- `src.data.dataset.CipherEpisode`: 한 문장 episode와 그 cipher table 소속
- `src.data.dataset.split_by_cipher_table`: 누수 없는 `f` 단위 분리
- `src.models.decoder.NeuralCipherDecoder`: 모든 baseline의 공통 모델
- `src.training.evaluate.evaluate_model`: token/zone/exact/top-k/길이별/unseen-f 지표
- `src.analysis.attention_debug.inspect_attention`: head별 attention 이웃 분석

초성 label 순서는 `ㄱ ㄲ ㄴ ㄷ ㄸ ㄹ ㅁ ㅂ ㅃ ㅅ ㅆ ㅇ ㅈ ㅉ ㅊ ㅋ ㅌ ㅍ ㅎ`입니다.

## Controlled synthetic benchmark

실제 한국어 corpus 없이 공통 Markov zone language와 table별 독립적인
zone→numeric-region permutation을 사용해 다섯 모델을 비교할 수 있습니다.

```bash
# 수 초 안에 전체 pipeline, 5개 baseline, A-D ablation을 확인
python -m src.benchmark.runner --config configs/synthetic_benchmark_smoke.yaml

# 100/20/20개 table, 5 epochs의 중간 sanity benchmark
python -m src.benchmark.runner --config configs/synthetic_benchmark_small.yaml

# 1600/200/200개의 train/validation/IID-test table, 3 seeds
python -m src.benchmark.runner --config configs/synthetic_benchmark.yaml
```

전체 설정은 추가로 200개의 numeric-support-shifted OOD table을 생성합니다.
모든 split은 서로 다른 `table_id`와 plaintext sequence를 사용하며, transition
matrix만 공유합니다. 기본 locality noise로 학습한 모델을 `0.0, 0.1, 0.25,
0.5` noise에서 다시 평가합니다. noise는 region width 대비 Gaussian 표준편차이며,
table 생성 시 각 `(zone, symbol)`에 한 번만 적용됩니다. 이후 암호화는 고정 lookup만
사용하며 기본적으로 전체 mapping의 cipher-value collision을 허용하지 않습니다.
강한 noise에서는 일부 mapping이 region 경계를 넘으므로 locality 자체가 점차 약해집니다.

부분 관측 Sinkhorn은 `R×19` real score에 `19-R`개의 neutral dummy row를 붙여
`19×19`로 정규화합니다. table-level 평가는 같은 `table_id`의 모든 occurrence
score를 region별로 평균한 뒤 row argmax와 Hungarian assignment를 각각 보고합니다.
전체 19개 region이 관측되고 모두 맞은 경우에만 table exact mapping으로 계산합니다.

결과 디렉터리에는 다음 파일이 생성됩니다.

- `results.json`: seed별 전체 metric, Sinkhorn 진단, 길이/noise 및 attention-distance 통계
- `results.csv`: 모델 및 seed별 비교 행
- `summary.md`: 평균 ± 표준편차 비교표
- `checkpoints/`: 각 모델과 seed의 best-validation checkpoint

`include_ablations: true`이면 relational baseline(A)에 더해 absolute digits와
cipher-relative feature를 켜고 끈 B/C/D 조건도 실행합니다. `relative OFF`는
digit/cipher delta만 제거하며 공통 sequence-position 관계는 유지합니다.

`use_absolute_sequence_position`은 sinusoidal input encoding을,
`use_relative_sequence_position`은 relational attention의 sequence delta를
각각 독립적으로 제어합니다. 이전 `use_sequence_position` 설정도 두 값을 함께
끄고 켜는 legacy alias로 계속 지원합니다.

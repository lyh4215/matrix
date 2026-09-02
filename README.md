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

## Google Colab

T4에서 128-token ciphertext의 train-table learning curve를 실행하려면 새 Colab
노트북에서 GPU runtime을 선택한 뒤 다음 셀을 실행합니다.

```python
!git clone https://github.com/lyh4215/matrix.git
%cd matrix
!pip install -e .
```

GPU가 연결되었는지 먼저 확인할 수 있습니다.

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

두 train 크기와 seed 하나로 실행 경로를 먼저 확인합니다. `--quick`도 실제 학습을
수행하므로 T4 사용을 권장합니다. quick 결과는 본 실험과 겹치지 않도록
`artifacts/learning_curve_128_quick/`에 저장됩니다.

```bash
!python learning_curve.py --quick
```

본 실험과 주요 실행 옵션은 다음과 같습니다.

```bash
!python learning_curve.py
!python learning_curve.py --batch-size 16
!python learning_curve.py --epochs 20
!python learning_curve.py --resume
```

기본값은 train table `50, 100, 200, 400, 800, 1600`, seed `41, 42, 43`,
`standard`/`relational`, 10 epochs, batch size 8입니다. 모든 episode는 길이 128이며
table마다 서로 다른 episode 두 개를 사용합니다. validation은 100 tables, IID test와
numeric-relocated OOD test는 각각 200 tables로 모든 train-size 조건에서 고정됩니다.
T4 메모리에 여유가 있으면 batch size를 `8 → 16 → 32` 순서로 올려볼 수 있습니다.

결과는 `artifacts/learning_curve_128/`에 저장됩니다. `raw_results.json`과
`results.csv`에는 run별 metric과 class-collapse 진단이, `learning_curve.csv`에는
plot용 tidy data가 들어갑니다. `summary.md`, train/IID/OOD PNG, run별
`training_history/`, relational `attention_statistics/`, `checkpoints/`도 함께 생성됩니다.
각 run이 끝날 때 바로 저장되며, 동일한 옵션으로 `--resume`을 주면 완료된 run을
건너뜁니다.

## Fixed-f overfit sanity check

unseen-f 실패가 기본 학습 pipeline 문제인지 permutation 일반화 문제인지 구분하기
위해, 하나의 deterministic cipher table을 train/validation/test에서 공유하는 sanity
benchmark를 제공합니다. plaintext episode만 split마다 새로 생성됩니다.

```bash
# 기본: train/validation/test 256/64/64 episodes, 길이 128, 50 epochs
python sanity_overfit.py

# 동일한 8개 episode를 dropout/weight decay 없이 기본 200 epochs 반복 학습
python sanity_overfit.py --tiny-memorize

python sanity_overfit.py --epochs 100
python sanity_overfit.py --train-episodes 512 --batch-size 16
```

시작할 때 PyTorch/CUDA/GPU/device가 출력되며 CUDA가 있으면 자동으로 사용합니다.
매 epoch의 train/validation loss·accuracy·prediction entropy·최대 예측 class 비율과
첫 optimizer step의 encoder/classifier gradient norm 및 parameter delta를 기록합니다.

기본 결과는 `artifacts/sanity_overfit/`, tiny 결과는
`artifacts/sanity_overfit_tiny/`에 저장됩니다. 두 모델의 history JSON, 고정 table
mapping, prediction distribution이 포함된 summary JSON/Markdown, checkpoint,
accuracy/loss curve PNG가 생성됩니다. summary의 성공/실패 문구는 90% 기본 threshold를
사용하는 heuristic이며 원인에 대한 확정 판정은 아닙니다.

## Absolute-number translation ablation

fixed-f 성능이 특정 절대 숫자 범위 암기에 의존하는지 확인하기 위해, 같은 plaintext와
underlying cipher mapping을 original/translated 조건에서 비교합니다. Translated 조건은
episode마다 하나의 seeded random integer offset을 모든 token에 동일하게 더합니다.
offset은 음수/양수 방향을 모두 샘플링하며 모든 결과를 `0..9999` 안에 유지하므로
pairwise cipher difference와 local geometry는 정확히 보존됩니다.

```bash
# original + translated, Standard + Relational
python translation_ablation.py

# translated 조건만
python translation_ablation.py --condition translated

# translated 조건을 100 epochs 학습
python translation_ablation.py --condition translated --epochs 100
```

기본값은 길이 128, train/validation/test `256/64/64` episodes, seed 42, 50 epochs,
batch size 8입니다. 결과는 `artifacts/translation_ablation/`의 raw/summary JSON과
Markdown, condition/model별 history와 checkpoint, validation accuracy PNG에 저장됩니다.
summary에는 Standard/Relational translation drop과 translated 조건의 모델 간 test
accuracy 차이가 포함됩니다. 동일 plaintext를 여러 valid offset으로 옮긴 hidden-state
cosine similarity 및 prediction consistency도 보조 지표로 기록합니다.

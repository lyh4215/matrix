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
- `relational_gated`: shared same-region gate + local/cross relational branches + token classifier
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

## Same-region gated attention and grouping probe

`relational_gated`는 기존 `relational`을 변경하지 않는 별도 모델입니다. Head-shared
same-region gate는 unsigned cipher distance, digit equality/absolute difference, symmetric
hidden-pair context를 사용하되 signed global numeric ordering은 받지 않습니다. Local branch는
signed numeric/digit delta를 사용할 수 있고, cross branch는 hidden-pair와 sequence relation만
사용하며 cipher/digit numeric delta를 받지 않습니다. Semantic 학습에서는 padding/self pair를
제외한 same-region BCE를 positive/negative class별로 평균해 `lambda_same_region` 가중치로
zone loss에 더합니다.

Permutation-invariant grouping만 먼저 검사하려면 다음을 실행합니다.

```bash
python same_region_probe.py \
  --models standard relational relational_gated \
  --train-tables 50 \
  --validation-tables 20 \
  --test-tables 50 \
  --epochs 50 \
  --batch-size 8 \
  --seed 42
```

결과는 `artifacts/same_region_probe/`에 저장됩니다. F1/balanced accuracy, distance bucket,
nearby cross-region hard negative, far within-region hard positive, gate separation과 validation에서
선택한 pure distance-threshold baseline을 함께 기록합니다.

19-way unseen-f semantic benchmark에서 gated 모델만 실행할 수도 있습니다.

```bash
python learning_curve.py \
  --models relational_gated \
  --train-table-counts 50 \
  --seeds 42 \
  --epochs 100 \
  --output-dir artifacts/learning_curve_gated_50f
```

## Oracle region-to-zone matching probe

`region_zone_match_probe.py`는 synthetic generator의 true numeric-region grouping만
oracle로 사용하고, 각 table의 region을 first-occurrence 순서로 다시 익명화합니다.
Matcher 입력은 anonymous transition graph와 frequency뿐이며 cipher value, digit,
numeric region 순서는 사용하지 않습니다. Generator가 반환한 공통 Markov transition
matrix를 canonical semantic structure로 직접 재사용합니다.

128-token oracle transition matching:

```bash
python region_zone_match_probe.py \
  --matchers oracle_transition \
  --sequence-lengths 128 \
  --test-tables 200 \
  --seed 42
```

Sequence-length study:

```bash
python region_zone_match_probe.py \
  --matchers oracle_transition \
  --sequence-lengths 64 128 256 512 \
  --test-tables 200 \
  --seed 42
```

Permutation-equivariant graph matcher 학습:

```bash
python region_zone_match_probe.py \
  --matchers learned \
  --train-tables 1600 \
  --validation-tables 100 \
  --test-tables 200 \
  --sequence-lengths 128 \
  --epochs 50 \
  --seed 42
```

구조 일관성으로 soft permutation을 반복 보정하는 별도 matcher:

```bash
python region_zone_match_probe.py \
  --matchers learned_structural \
  --structural-refinement-steps 4 \
  --structural-beta 0.1 \
  --lambda-graph 0.5
```

`learned_structural`은 기존 `learned` GNN/unary 구현을 상속하고,
`S0 = Sinkhorn(U)` 뒤에 `R = C S (log P)^T + C^T S log P`,
`S_next = Sinkhorn(U + beta R)`를 기본 4회 반복합니다. `C`는 smoothing하지 않은
anonymous transition count, `P`는 `bundle.transition_matrix`이며 checkpoint buffer에
저장됩니다. `log P`는 `clamp_min(1e-8)`로 계산하고 R의 행 평균을 빼서 logit 크기를
줄입니다(Sinkhorn 결과는 동일). Count를 정규화하지 않으므로 beta는 transition 한 건당
가중치이며 sequence 길이/개수가 커질수록 구조 점수의 영향도 커집니다.

학습 loss는 기존 observed permutation NLL에 `lambda_graph * Lgraph`를 더합니다.
`Q_hat = S P S^T`, `Lgraph = -sum(C * log(clamp(Q_hat, 1e-8))) / sum(C)`를
graph별로 계산해 batch 평균하며, transition이 없는 graph는 0을 기여합니다.
미관측 region은 기존 neutral-dummy Sinkhorn으로 처리하고, 실제 count가 없는 edge는
loss에 기여하지 않습니다. 반복 과정 전체가 미분 가능하며 numeric cipher value,
digit, numeric region 위치를 추가 입력하지 않습니다.

YAML의 `learned_structural` 항목에서 `structural_refinement_steps`(기본 4),
`structural_beta`(0.1), `lambda_graph`(0.5)를 설정합니다. GNN 크기와 optimizer/epoch
설정은 기존 `learned` 항목을 공유하며, 두 matcher는 각각 독립적으로 학습합니다.
`--matchers learned learned_structural`로 함께 실행할 수 있고, 새 모델의 history와
checkpoint는 `learned_structural/`에 별도로 저장합니다. Steps 또는 beta를 0으로
설정하면 unary Sinkhorn만 사용하며 graph loss는 lambda가 0일 때 비활성화됩니다.
Structural refinement의 Sinkhorn은 batch 전체를 한 번에 계산합니다. 미관측 region을
제자리 neutral dummy row로 두어 기존 Sinkhorn과 같은 확률/gradient를 유지하면서
GPU에서 batch의 각 예제마다 반복 연산을 별도로 호출하는 부담을 줄입니다.

학습된 permutation을 초기값으로 count-NLL local search를 추가하려면
`--local-search`를 지정합니다. 기존 두 모델의 학습과 예측은 그대로 남고,
`learned_local_search`와 `learned_structural_local_search` 결과가 별도로 추가됩니다.
검색은 CPU에서 기존 oracle의 delta-scored pair swap/3-cycle descent를 재사용합니다.
기본 초기값은 learned 예측 하나이며 oracle 초기화나 정답 label은 사용하지 않습니다.
`--local-search-max-iterations`(기본 50), `--local-search-restarts`(기본 1)를 조절할 수
있습니다. 추가 restart는 learned permutation의 1~3개 pair를 바꾼 초기값을 사용합니다.
항상 보정 전 permutation을 후보로 유지하고 최종 count NLL을 다시 계산하므로
초기 예측보다 NLL이 나쁜 결과는 반환하지 않습니다. YAML 항목은 `local_search`입니다.

`summary.md`에는 같은 table의 보정 전후 assignment/token delta, assignment 정확도가
개선/악화/동일한 table 수, transition당 NLL 감소량, table당 검색 시간과 수렴률을
기록합니다. 검색 시간은 neural inference와 데이터 생성 시간을 제외합니다.
NLL이 좋아져도 finite-sample noise 때문에 정답률은 나빠질 수 있으므로 둘을 함께
비교해야 합니다. Validation checkpoint 선택은 보정 전 accuracy를 그대로 사용합니다.

기존 실험의 `raw_results.json`이 있으면 **재학습이나 checkpoint 로딩 없이** 보정할 수
있습니다. 원본 config/seed로 test graph를 다시 만들고 저장된 예측을 초기값으로 씁니다.
원본 결과와 canonical matrix/table identity가 맞는지도 확인합니다.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python region_zone_refine.py \
  --results artifacts/region_zone_match_probe/raw_results.json \
  --output-dir artifacts/region_zone_match_refined \
  --restarts 1
```

원본 `raw_results.json`이 있는 디렉터리는 덮어쓸 수 없습니다. 새 결과 디렉터리에
원래 baseline과 보정 결과를 함께 저장합니다. 이전 버전의 `learned` 결과도 지원하며,
정답률 비교용 `summary.json`만으로는 개별 table 예측이 없어 실행할 수 없습니다.

Colab에서 새 실험을 실행하려면
[준비된 notebook](https://colab.research.google.com/github/lyh4215/matrix/blob/main/notebooks/region_zone_structural_colab.ipynb)을
열어 GPU 런타임으로 실행하세요. 기본값은 길이 128/256 **각각 별도 학습**, train 1600 /
validation 100 / test 200 tables, 50 epochs, batch 8, seed 42입니다. 두 모델과 각각의
local search를 같은 데이터에서 비교하고, 결과/로그/checkpoint를 Drive에 저장해 ZIP으로
묶습니다. 먼저 128만 실행하려면 `LENGTHS = [128]`로 바꾸세요. 기존 raw 결과가 있다면
`SOURCE_RESULTS`에 경로를 지정해 학습을 생략할 수 있습니다. `INCLUDE_ORACLE = True`는
같은 test tables의 oracle 결과도 재계산하며, 이 탐색은 GPU로 가속되지 않습니다.

한 table에서 합칠 sequence 수는 `--sequences-per-table N`으로 조절합니다. 결과는
기본적으로 `artifacts/region_zone_match_probe/`에 raw/summary JSON과 Markdown,
oracle table diagnostics, learned history/checkpoint, length curve와 confusion matrix로
저장됩니다. Observed/full assignment accuracy, token reconstruction accuracy, coverage,
exact recovery, empirical-Q true-alignment objective와 canonical signature ambiguity를 함께
기록합니다.

Oracle 기본 objective는 empirical transition count likelihood인 `count_nll`이며,
`--oracle-objective mse`로 기존 row-normalized MSE를 선택할 수 있습니다. Summary에는
predicted/true-permutation objective와 gap의 mean/median, predicted objective가 true 이하인
table 비율이 포함됩니다.

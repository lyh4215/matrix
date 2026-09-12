# 한국어 trigram 점수 진단 (재학습 없음)

기존 Korean corpus benchmark ZIP을 입력받아, **정답 permutation이 기존 오답 후보보다 낮은 NLL을 받는지** 확인한다. trigram으로 새 permutation을 탐색하는 단계는 아직 실행하지 않는다.

Colab에서 저장소를 갱신하고 기존 ZIP 경로를 지정한다. CPU 런타임으로도 실행할 수 있다.

```python
%cd /content/matrix
!git pull --ff-only
!pip install -q -e .
!python korean_trigram_diagnostic.py --results /content/20260912_004050_972164_UTC.zip
```

ZIP이 다른 경로에 있다면 `--results`만 바꾼다. 실행 마지막에 나오는 Results ZIP을 공유하면 된다. 원래 실험을 다시 학습하거나 GPU에서 50 epochs를 돌릴 필요는 없다. validation 후보 생성에는 CPU oracle/local search 시간이 필요하다.

## 무엇을 비교하는가

기존 train 문단만으로 초성 trigram count `N[a,b,c]`를 계산한다. 문서 분할, 정규화, 중복 제거, 문단 경계는 기존 corpus benchmark와 동일하다. 문단 사이의 triple은 세지 않는다. 이전 ZIP의 corpus hash, window manifest, split, canonical P가 재구성 결과와 같은지 검사한다.

기존 train bigram 확률을 `P[c|b]`라 할 때:

```text
T[c|a,b] = (N[a,b,c] + k * P[c|b]) / (sum_c N[a,b,c] + k)
M[c|a,b] = (1-w) * P[c|b] + w * T[c|a,b]
NLL = -log P[z1|z0] - sum(t=2..L-1) log M[zt|z(t-2),z(t-1)]
```

`k`는 bigram으로 되돌아가는 smoothing 강도다. 관측되지 않은 history에서는 정확히 bigram으로 되돌아간다. `w=0`은 기존 count-bigram NLL과 같다. unigram 항이나 새로운 신경망은 추가하지 않는다.

- 고정 grid: bigram baseline (`w=0`), `k ∈ {1,10,100}` × `w ∈ {0.25,0.5,0.75,1}`.
- validation: 기존 structural checkpoint로 추론 후 기존 bigram local search를 적용하고, oracle도 생성한다. 두 후보를 모든 점수 설정에서 동일하게 사용한다.
- 선택 기준: validation에서 **모든 오답 후보보다 정답 NLL이 엄격히 낮은 구간 비율** 최대. 동률이면 validation 후보 선택 token accuracy가 높은 설정, 그래도 같으면 grid에서 먼저 나온 설정을 사용한다.
- test: 선택을 확정한 후 **bigram baseline과 선택된 설정만** 평가한다. ZIP에 저장된 oracle/structural-local-search 후보를 사용한다. 정답은 후보로 추가하지 않는다. 후보 NLL 동률은 oracle을 선택한다.

관측된 region에서 정답인 후보는 오답 집합에서 제외한다. 오답 후보가 하나도 없는 구간은 truth-win 비율의 분모에서 제외한다. NLL 차이의 절댓값이 `1e-8` 이하이면 진단상 동률로 집계한다.

## 결과 해석

`summary.md`에 validation grid와 test의 truth-win 수, assignment/token 정확도를 표시한다. `diagnostic.json`은 정답 NLL, 각 후보 NLL, 선택된 후보, 구간별 정오를 포함한다. `validation_candidates.json`은 생성한 후보를 기록한다.

- truth-win 상승: 기존 오답을 구별하는 점수가 개선됐다는 근거다.
- 후보 선택 정확도 상승: 두 후보 중 더 좋은 것을 고르는 데 도움이 됐다는 근거다.
- 둘 다 오답인 경우, 후보 선택만으로 정답을 만들 수 없다.
- 여기서 좋아져도 trigram local search가 잘 작동한다는 보장은 없다. 이 진단을 보고 다음 탐색 실험을 결정한다.

validation은 이미 신경망 checkpoint 선택에 사용된 split이다. test 결과도 이전에 확인했던 데이터이므로 완전히 새로운 blind test는 아니다. 두 후보에 대한 진단이며 전체 permutation 공간에서의 식별 가능성이나 성능 상한을 측정하지 않는다. 초성 zone 복원 실험이라는 범위도 그대로다.

## 작은 실행 확인

```bash
python korean_trigram_diagnostic.py --results /path/to/benchmark.zip --smoke
```

validation/test 각각 처음 2개 구간과 축소한 validation oracle 탐색만 사용한다. 결과에 `smoke: true`를 기록하며 본 실험 결론에 사용하지 않는다. 원본 corpus가 로컬 파일이었다면 동일한 파일/디렉터리를 `--corpus`로 지정해야 한다.

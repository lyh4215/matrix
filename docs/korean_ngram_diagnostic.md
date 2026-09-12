# 3~7-gram 한국어 점수 비교

기존 benchmark ZIP에서 checkpoint와 test 후보를 읽어 사용한다. 새 epoch 학습이나 n-gram local search 없이, 고정된 oracle/structural-local-search 후보를 평가한다. CPU 런타임으로 실행할 수 있다.

```python
%cd /content/matrix
!git pull --ff-only
!pip install -q -e .
!python korean_ngram_diagnostic.py --results /content/20260912_004050_972164_UTC.zip
```

`--results`에는 **원래 Korean corpus benchmark ZIP**을 지정한다. 직전 trigram 진단 결과 ZIP이 아니다. `--max-order 4`로 4-gram까지만 비교할 수 있다. 기본값은 7이다. `--max-order 5`로 이전 비교 범위를 유지할 수 있다. `--smoke`는 validation/test 각각 2구간으로 실행만 확인한다. `--cache-dir`, `--corpus`, `--output-dir`도 기존 진단과 동일하게 지원한다.

## 점수와 설정 선택

원래 train 문단만으로 3~7-gram count를 센다. NFC 정규화, 한글 음절 추출, train 문단 중복 제거와 문서 분할은 기존 benchmark와 같다. 문단 경계를 넘어 count를 만들지 않는다. corpus hash, split, window manifest와 canonical P를 재현 검증한다.

```text
P2 = 기존 train bigram 확률
Pn(c | history) = [N(history,c) + k * P(n-1)(c | suffix(history))]
                 / [N(history) + k]
score probability = (1-w) * P2 + w * Pn
```

각 설정의 smoothing 강도 `k`는 3부터 해당 차수까지 공통으로 적용된다. 긴 history가 없으면 한 단계 짧은 확률로 돌아가며, 짧은 확률도 같은 방식으로 추정된다. 시퀀스 시작에서는 확보된 길이만큼의 문맥을 사용한다. 첫 초성 자체의 unigram 항은 포함하지 않는다. `w=0`은 원래 bigram NLL, order=3은 기존 trigram 진단과 같은 수식이다.

validation에서 비교하는 grid는 bigram 1개와 `order ∈ {3,4,5,6,7}` × `k ∈ {1,10,100}` × `w ∈ {0.25,0.5,0.75,1}`의 **총 61개**다. 모든 설정은 같은 두 후보를 평가한다.

선택 기준은 기존과 같다: 정답 NLL이 모든 오답 후보보다 낮은 비율, 동률이면 후보 선택 token accuracy. 그마저 같으면 낮은 차수, 작은 k, 작은 w 순서를 따른다. 오답이 없는 구간은 truth-win 분모에서 제외한다. 정답은 후보로 추가하지 않는다.

**선택을 확정한 뒤 test에서는 다음만 평가한다.**

- bigram baseline
- 미리 고정한 이전 trigram 기준 (`order=3, k=1, w=1`)
- 미리 고정한 이전 5-gram 기준 (`order=5, k=100, w=1`; max-order가 5 이상일 때)
- validation에서 선택한 설정 (중복이면 한 번만 평가)

test에서 차수나 smoothing을 다시 선택하지 않는다.

## 희소성 진단

`summary.md`에는 validation grid의 정답 선호, assignment/token 정확도와 함께 각 차수·강도의 `w=1` 희소성 표를 출력한다. `diagnostic.json`에는 validation 전체 설정 및 평가한 test 설정의 세부 값이 포함된다. 정답 sequence와 두 후보가 복원한 sequence를 **각각 따로** 집계한다.

- `unseen_history_rate`: 해당 n-gram의 앞 n−1개 초성을 train에서 관측하지 못한 위치 비율.
- `unseen_ngram_rate`: 다음 초성까지 포함한 n개 패턴을 train에서 관측하지 못한 위치 비율.
- `mean_lower_order_mass`: `k/(N(history)+k)`의 평균. 한 단계 짧은 모델에 의존하는 정도다. 외부 interpolation weight `w`까지 합친 최종 bigram 비중은 아니다.

해당 길이의 history가 있는 위치만 분모에 포함한다. 구간 평균이 아니라 위치 수 기준 평균이다. 이 값들은 추가 설명용이며 설정 선택 기준으로 사용하지 않는다.

출력되는 Results ZIP을 공유하면 된다. validation은 기존 checkpoint 선택에도 사용됐고 test도 앞선 실험에서 이미 살펴본 데이터이므로, 결과는 완전히 새로운 blind test가 아니다. 두 고정 후보에 대한 점수 진단이며, 전체 permutation에서의 탐색 성능이나 완전한 한글 복호화 정확도를 뜻하지 않는다.


## 6·7-gram 저장 및 해석

최대 차수가 6 이상이면 모든 차수에 sparse count 저장을 사용한다. 관측된 패턴을 base-19 정수 key로 저장하고 history count도 관측된 것만 보관한다. 19⁷ 크기의 count/확률 배열은 만들지 않는다. 평가 위치에서 필요한 확률만 짧은 차수부터 계산한다. 메모리 사용은 실제 관측 패턴 수에 따라 증가한다. 결과 JSON의 `observed_patterns`에서 차수별 저장 패턴 수를 확인할 수 있다.

`mean_mass_to_order_5_or_lower`는 6·7차의 backoff 비중을 해당 위치에서 연속으로 곱한 값의 평균이다. 예를 들어 7-gram에서는 7→6 비중 × 6→5 비중이다. 값이 1에 가까우면 높은 차수에서 대부분 5-gram 이하로 돌아간다. 직접 bigram과 섞는 외부 weight는 이 통계에 포함하지 않는다. 기본 표는 weight=1 설정에서 이를 보여준다. 각 차수의 충분한 문맥이 확보된 위치만 계산하며, 문장 시작부의 짧은 문맥은 제외한다.

차수 증가 효과는 validation에서 확인하고 설정을 고정한다. test에서 6·7-gram 모든 설정을 반복 비교하지 않는다. 기존 두 후보가 모두 오답이면 재평가만으로 정답을 만들어낼 수 없다는 한계는 동일하다.

# Oracle / structural complementarity experiment

기존 128-token 실험을 재학습 없이 비교하려면 repository를 업데이트한 뒤 실행합니다:

```bash
python region_zone_complementarity.py \
  --results artifacts/region_zone_benchmark/20260908_132212_329737_UTC/length_128/raw_results.json
```

실제 저장 위치에 맞춰 `--results`만 바꾸면 됩니다. 통합 `summary.json`이 아닌
**길이별 `raw_results.json`**이 필요합니다. GPU나 checkpoint 로딩, 추가 epoch 학습은
없습니다. 원본 config/seed로 test graphs를 재생성하고 저장된 learned 예측에서
local search를 다시 실행합니다. Oracle도 같은 graphs에서 다시 계산합니다.
기존 실험의 local-search 설정을 그대로 사용합니다.

기본 출력은 원본 파일 옆 `oracle_comparison/`과 `oracle_comparison.zip`입니다.
`--output-dir`로 변경할 수 있습니다. 원본 결과는 보존합니다.

새 학습까지 포함하려면:

```bash
python region_zone_benchmark.py --sequence-lengths 128 --compare-oracle
```

`--compare-oracle`은 oracle 실행과 local search를 활성화하고
`oracle_structural_nll_select` 평가 결과를 추가합니다. 기존 모델 학습은 그대로입니다.

## 선택 규칙

각 table에서 `oracle_transition`과 `learned_structural_local_search`의 permutation을
동일한 raw count C, canonical P, epsilon으로 다시 평가합니다.
`-sum(C_ij * log(P[pi(i), pi(j)]))`가 더 작은 후보를 선택합니다.
NLL이 정확히 같으면 oracle을 유지합니다. 선택 과정은 정답 label을 받지 않습니다.

## 확인할 결과

- 두 후보와 NLL selector의 assignment/token/exact recovery accuracy.
- 동일한 observed permutation을 찾은 table 수.
- Exact recovery 교집합: both / oracle only / structural only / neither.
- 각 후보 대비 selector가 정확도를 개선/악화시킨 table 수.
- Oracle / structural / selected 각각의 `NLL <= true NLL` 비율, mean/median gap.
- Label-assisted best-of-two: 정답을 알고 table마다 좋은 후보를 골랐을 때의 진단용
  상한. 실제 selector 성능과 구분하며 선택에는 사용하지 않습니다. Assignment와 token의
  상한은 각 지표별로 별도 계산하므로 하나의 실제 선택 정책을 나타내지 않습니다.
- NLL selector가 더 정확한 후보를 놓친 table 수.

NLL 감소가 정답률 상승을 보장하지 않으며, `NLL <= true` 역시 전역 최적의 증명이 아닙니다.
상세 table별 NLL, gap, 선택 출처와 양쪽 permutation은 `raw_results.json`에 저장합니다.

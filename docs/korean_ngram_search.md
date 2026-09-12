# 한국어 5/6-gram local search

기존 두 후보 중 하나를 고르는 데서 나아가, 각 후보의 permutation 자체를 n-gram NLL로 개선한다. 주 실험은 이전 validation에서 선택된 **6-gram / strength=100 / weight=1**이고, **5-gram / strength=100 / weight=1**을 동일 조건의 비교 기준으로 실행한다. test 결과로 설정을 다시 고르지 않는다.

```python
%cd /content/matrix
!git pull --ff-only
!pip install -q -e .
!python korean_ngram_search.py --results /content/20260912_004050_972164_UTC.zip
```

입력은 **원래 Korean corpus benchmark ZIP**이다. 진단 ZIP이 아니다. CPU에서 실행하며 새 epoch 학습이나 checkpoint 추론은 없다. 원본 corpus를 재구성하고 train 문단의 sparse n-gram count를 추정한 다음, 저장된 test 후보로 바로 탐색한다. corpus hash, 문서 분할, window와 canonical P가 기존 결과와 일치하는지 검증한다.

## 탐색

각 test 구간에서 `oracle_transition`, `learned_structural_local_search`의 저장된 assignment를 각각 시작점으로 사용한다. 각각의 탐색은 다음 순서다.

1. 모든 pair swap을 검사하고 가장 큰 NLL 감소를 주는 이동을 수락한다.
2. 개선 가능한 swap이 없다면 모든 3-cycle의 양방향을 검사하고 최선의 개선을 수락한다.
3. 이동했으면 1로 돌아간다. 두 종류 모두 개선이 없으면 지역 최적점으로 판정한다.
4. 기본 최대 50회 이동에 도달하면 종료하되 수렴했다고 표시하지 않는다.

미관측 region도 관측 region과 교환 가능하다. 미관측 region끼리만 바꾸는 무의미한 이동은 생략한다. 모든 이동은 일대일 permutation을 보존하고, NLL 감소가 `1e-8`보다 클 때만 수락한다. 정답은 탐색이나 시작점 선택에 입력하지 않는다.

두 시작점의 탐색이 끝나면 같은 차수의 NLL이 낮은 결과를 선택한다. 정확한 동률은 oracle에서 시작한 결과를 유지한다. 최초 assignment는 각 탐색의 fallback이므로 해당 목적함수가 나빠지는 결과를 반환하지 않는다. **NLL 감소가 정확도 상승을 보장하지는 않는다.**

## 실행량과 결과

각 차수에서 test 200개 × 시작점 2개를 탐색하므로 앞선 고정 후보 진단보다 오래 걸린다. `--batch-size 128`은 이웃 permutation을 묶어 평가하는 크기다. 학습 batch가 아니며 값이 커지면 메모리 사용이 늘 수 있다. `--max-moves 50`은 시작점별 수락 이동 제한이다. 탐색 설정을 바꾼 결과는 별도 실험으로 취급한다.

```bash
# 작은 실행 확인: test 2개, 시작점별 최대 2회 이동
python korean_ngram_search.py --results /path/to/benchmark.zip --smoke
```

`--corpus`, `--cache-dir`, `--output-dir`도 지원한다. Google Drive에 저장하려면 마운트 후 `--output-dir /content/drive/MyDrive/matrix-ngram-search`를 지정한다. 구간마다 진행 상황을 출력하고 `table_results.jsonl`에 기록한다. 자동 재개 기능은 없다.

최종 ZIP에는 다음 결과가 들어간다.

- `before_select`: 기존 두 후보 중 해당 n-gram NLL로 선택한 결과.
- `oracle_transition`: oracle 시작점에서 **n-gram 탐색 후** 결과.
- `learned_structural_local_search`: structural-local-search 시작점에서 **n-gram 탐색 후** 결과.
- `after_select`: 두 탐색 결과 중 해당 n-gram NLL로 선택한 결과.

`summary.json`/`summary.md`는 assignment/token/exact 정확도, 두 시작점을 합친 구간당 탐색 시간, 시작점별 수렴률과 수락 이동 수, 탐색 전 대비 정확도 개선·악화 구간 수를 제공한다. `table_results.jsonl`은 시작점별 최종 permutation, NLL 변화, 이동 종류, 수렴 여부, 정답 NLL과 정확도 진단을 포함한다. 시간은 corpus 준비를 제외한 탐색 시간이다.

완료 후 출력되는 **Results ZIP**을 공유하면 된다. 수렴은 swap/3-cycle 지역 최적점만 뜻하며 전역 최적성을 보장하지 않는다. test는 이전에 살펴본 자료이고, 초성 zone 복원 실험이라는 범위도 유지된다.

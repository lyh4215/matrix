# 실제 한국어 corpus의 256음절 region→zone benchmark

Colab에서 GPU 런타임을 선택한 뒤 다음 셀을 실행한다. `/content/matrix`에 저장소가 이미 있다는 기준이다.

```python
%cd /content/matrix
!git pull --ff-only
!pip install -q -e .
!python korean_corpus_benchmark.py
```

처음 실행하면 약 39 MB의 KorQuAD 원본 JSON을 자동으로 내려받고 SHA256을 검증한다. 이후에는 `artifacts/corpus_cache`의 캐시를 사용한다. 저장소가 없다면 먼저 `!git clone https://github.com/lyh4215/matrix.git /content/matrix`를 실행한다.

## 데이터와 실험 범위

- 출처: [KorQuAD 1.0 공식 페이지](https://korquad.github.io/category/1.0_ENG.html), [공식 train JSON](https://korquad.github.io/dataset/KorQuAD_v1.0_train.json). Wikipedia에서 수집된 한국어 문단을 사용하며, 질문과 정답은 사용하지 않는다. 배포 라이선스는 CC BY-ND 2.0 KR이다.
- 원본 SHA256: `40d5115879a701751781df721d901abfa736d8db5f89000f2619433f39bf2dd2`.
- 원본 train 파일의 문서 제목을 기준으로 새롭게 80/10/10% 분리한다. 같은 문서의 문단은 서로 다른 split에 들어가지 않는다. 공식 KorQuAD QA 평가와는 별개의 실험이다.
- NFC 정규화 후 완성형 한글 음절만 남기고 공백·문장부호·숫자·다른 문자는 제거한다. **256 tokens는 한글 256음절**이다. 각 문단 내부에서 겹치지 않는 구간을 추출하며 짧은 문단을 이어 붙이거나 패딩하지 않는다.
- 원문 음절 순서를 그대로 사용한다. Markov 모델에서 문장을 생성하지 않는다. 동일한 정규화 문단 및 동일한 256음절 구간은 중복 제거하지만 유사 문서까지 탐지하지는 않는다.
- canonical `P`는 **train 문단 전체**의 초성 전이 count에 행별 smoothing을 적용해 추정한다. 모델 학습에 선택된 구간 외의 train 문단도 포함한다. validation/test는 `P` 추정에 사용하지 않는다.
- 기존 matcher의 점수는 여전히 1차 전이 기반이다. 이번 실험은 실제 문장에서 그 방법이 얼마나 잘 작동하는지 확인한다.
- 기존 probe와 동일하게 정답 초성별 region grouping이 주어진 상태에서 익명 region→19개 초성을 복원한다. 숫자 암호값은 입력하지 않는다. **token accuracy도 초성 zone 정확도이며 완전한 한글 음절 복호화 정확도가 아니다.**

기본 seed 42에서 문서 수는 train 1,136 / validation 142 / test 142개이고, 사용 가능한 256음절 구간은 각각 7,253 / 1,123 / 697개다. 같은 split 안에서는 여러 구간이 한 문서에서 나올 수 있다.

## 기본 비교

`configs/korean_corpus_benchmark.yaml`의 기본 설정은 train 1,600 / validation 100 / test 200 구간, 50 epochs, batch size 8, CUDA다. `learned`와 `learned_structural`을 실제 corpus로 각각 새로 학습한다. local search와 후보 선택은 추가 학습 없이 수행한다.

결과에는 다음 8개 방식이 포함된다.

1. `random`
2. `frequency`
3. `oracle_transition`
4. `learned`
5. `learned_structural`
6. `learned_local_search`
7. `learned_structural_local_search`
8. `oracle_structural_nll_select`

여기서 `oracle_transition`도 train에서 추정한 `P`를 사용한다. 실제 데이터 생성 분포나 test의 정답 permutation을 제공받는 것은 아니다. 관측 region assignment, token accuracy, 관측 region exact recovery, coverage 및 oracle/structural 상호보완성 진단을 함께 확인한다. 실제 corpus에서는 희귀 초성이 빠질 수 있으므로 coverage를 반드시 함께 읽는다.

## 실행 옵션과 결과

```bash
# 데이터 다운로드·분할·coverage만 확인 (학습 및 oracle 탐색 없음)
python korean_corpus_benchmark.py --prepare-only

# 실제 corpus로 19 zones / 256음절 / 4·2·2 구간 / 1 epoch CPU smoke
python korean_corpus_benchmark.py --smoke

# 학습량 조정
python korean_corpus_benchmark.py --epochs 30 --batch-size 16

# Google Drive를 마운트했다면 결과를 직접 저장
python korean_corpus_benchmark.py --output-dir /content/drive/MyDrive/matrix-korean-results

# 본인 corpus: 문서마다 하나의 UTF-8 .txt 파일을 둔 디렉터리
python korean_corpus_benchmark.py --corpus /content/korean-documents
```

`--corpus`는 `{"id":"문서 ID","text":"문단 원문"}` 형식의 JSONL도 지원한다. 같은 ID는 같은 문서로 묶는다. corpus 크기가 작으면 `--train-tables`, `--validation-tables`, `--test-tables`를 줄여야 한다. 부족한 구간을 반복 복제하지 않는다. 구조 관련 설정은 YAML의 `learned_structural`에서 조정한다.

결과는 `artifacts/korean_corpus_benchmark/<UTC timestamp>/`와 같은 이름의 ZIP으로 저장된다. `summary.md`, `summary.csv`, `raw_results.json`, 모델 checkpoint, 그래프, `corpus_manifest.json`, `run_config.json`을 포함한다. manifest에는 출처·해시·분할·구간 위치·coverage·canonical count·환경을 기록한다. 원본 corpus JSON은 ZIP에 포함하지 않는다.

실행 후 출력된 **Results ZIP**을 공유하면 된다. 기존 `region_zone_refine.py` / `region_zone_complementarity.py`의 저장 결과 재실행은 synthetic 전용이므로 corpus 결과에는 사용할 수 없다. 이 benchmark 실행에 local search와 상호보완성 비교가 이미 포함되어 있다.

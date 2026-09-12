# 익명 region sequence Transformer 비교

기존 GNN `learned`와 새로운 `learned_sequence`를 같은 KorQuAD 문서 분할·구간·seed·optimizer·학습 epoch로 비교한다. **이번에는 두 모델 모두 새로 학습**하므로 Colab GPU 런타임을 선택한다.

```python
%cd /content/matrix
!git pull --ff-only
!pip install -q -e .
!python korean_sequence_benchmark.py
```

원문 자동 다운로드와 hash 검증은 기존 corpus benchmark와 같다. 이전 결과 ZIP은 필요 없다. 기본값은 256 한글 음절, train 1600 / validation 100 / test 200 구간, 50 epochs, batch 8, width 64, 3 layers다. 테스트 데이터의 범위는 이전 실험과 같으며 새로운 blind test가 아니다.

## 새 구조

```text
익명 region 등장 순서 [B,L]
→ 각 위치의 region 등장 빈도 + sinusoidal text-position encoding
→ 4-head self-attention × 3 layers
   attention logits += head별 학습 계수 × (두 위치가 같은 region인가)
→ 같은 region의 위치 표현을 평균 pooling
→ 19개 semantic prototype과 MLP compatibility
→ masked Sinkhorn → Hungarian 일대일 assignment
```

region ID는 equality 검사와 pooling에만 사용한다. ID별 embedding, 숫자 암호값, 숫자 region 위치는 입력하지 않는다. 위치 인코딩은 **문장 내 등장 위치**다. 그래서 region ID를 임의로 다시 붙여도 출력의 region 행만 같은 방식으로 재배열된다. 이 성질은 dropout을 끈 평가 모드에서 검사한다.

padding은 -1이며 attention key와 pooling에서 제외한다. 시퀀스 하나당 graph 하나만 지원한다. 서로 다른 문단을 이어 붙이지 않는다. 희귀 초성이 없는 경우에는 기존 matcher와 같은 masked Sinkhorn을 사용한다.

## 공정한 비교와 범위

두 모델 모두 observed permutation NLL만 최적화한다. **graph loss, structural refinement, n-gram 점수, local search는 이번 비교에 적용하지 않는다.** 새로운 sequence matcher는 canonical P도 입력받지 않는다. corpus 코드가 P를 생성하는 것은 기존 결과 형식 유지를 위한 것이다.

동일한 폭·층 수·학습 데이터·update 횟수를 사용하지만 파라미터 수와 FLOPs가 완전히 같은 비교는 아니다. 각 checkpoint에 `parameter_count`를 기록한다. Transformer는 256개 위치에 attention을 적용하므로 19개 region GNN보다 계산량이 크다. 성능 향상을 미리 가정하지 않고 동일 조건에서 확인한다.

checkpoint 선택은 validation observed assignment accuracy 기준이고 test는 선택된 checkpoint로만 평가한다. 현재 1600개 구간으로 충분한지는 아직 검증되지 않았다. 데이터 양을 늘리는 실험은 구조 비교 결과를 확인한 뒤 별도로 진행한다.

## 옵션과 결과

```bash
# CPU에서 실제 corpus 4/2/2 구간, 1 epoch 실행 확인
python korean_sequence_benchmark.py --smoke

# 두 모델 모두 같은 설정으로 조정
python korean_sequence_benchmark.py --epochs 30 --batch-size 8
```

기타 corpus runner 옵션(`--corpus`, `--cache-dir`, `--output-dir`, `--seed` 등)도 지원한다. 설정 파일은 `configs/korean_sequence_benchmark.yaml`이다. attention head 수는 4로 고정하며 width는 4의 배수여야 한다. 기존 probe CLI에서도 `--matchers learned_sequence`를 선택할 수 있다.

결과 ZIP에는 두 모델의 accuracy, confusion matrix, 구간별 assignment, 학습 history와 checkpoint가 포함된다. `learned_sequence/history.json` 및 `raw_results.json`의 `learned_sequence_history`로 학습·validation 추이를 확인할 수 있다. 실행 마지막의 **Results ZIP**을 공유하면 된다.

여전히 정답 region grouping이 주어진 **초성 zone 복원** 실험이다. 한글 음절 전체를 복호화하는 정확도와는 다르다.

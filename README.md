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

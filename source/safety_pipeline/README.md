# safety_pipeline — 모듈 구조 & 협업 가이드

안전신문고 이미지-텍스트 의미 정확성 평가 파이프라인을 두 명이 나눠 작업할 수 있도록
단계별 모듈로 분리했습니다.

## 실행 순서

```
Stage A (담당 A)                         Stage B (담당 B)
run_stage_a_extraction.ipynb    →  파일  →  run_stage_b_verification.ipynb
  LLM으로 claim 추출·합의               VLM으로 이미지 검증·채점
  stageA_claims__*.json 저장            stageA_claims__*.json 읽어 사용
```

두 단계는 **디스크의 claim 집합 파일(`stageA_claims__*.json`)** 을 유일한 접점으로 합니다.
따라서 A와 B는 서로 다른 시점/PC 에서, 서로 다른 파일만 수정하며 독립적으로 작업할 수 있습니다.

## 파일 소유

| 구분 | 파일 | 내용 |
|---|---|---|
| **공유** | `config.py` | 유형 체계, 유형 중요도, 비교 모델 목록, 파라미터 |
| **공유** | `common.py` | JSON 파싱/복구, 시드, dtype, id |
| **공유** | `records.py` | A↔B 데이터 계약 (`ClaimSet`, `EvaluationRecord`) |
| **공유** | `dataio.py` | 데이터 로드, claim 집합 저장/로드 |
| **공유** | `analysis.py` | 모델 비교 요약(+사람 정답 상관/오차) |
| 👤 **A** | `extraction.py` | 추출 프롬프트 + 텍스트 LLM 어댑터 |
| 👤 **A** | `consensus.py` | 임베딩 어댑터 + 의미 합의 |
| 👤 **A** | `stage_a.py` | Stage A 오케스트레이션 |
| 👤 **B** | `verification.py` | 검증 프롬프트 + VLM 어댑터 |
| 👤 **B** | `scoring.py` | 점수 집계 |
| 👤 **B** | `stage_b.py` | Stage B 오케스트레이션 |

> ⚠️ `config.py` 의 `CLAIM_TYPES` / `DEFAULT_TYPE_IMPORTANCE` 는 A·B 양쪽에 영향을 줍니다.
> 이 두 값을 바꿀 때만 서로 알리면, 나머지는 충돌 없이 병렬 작업 가능합니다.

## 데이터 계약 (seam)

`records.py` 상단 주석에 claim dict / verdict dict 스키마가 정의되어 있습니다.
A는 claim dict 를 **생산**하고, B는 그것을 **소비**해 verdict dict 를 생산합니다.

## 새 모델 추가해서 비교하기

- **A**: `config.CLAIM_MODELS` 또는 `EMBEDDING_MODELS` 에 항목 추가 → Stage A 가 조합별로 산출물 생성
- **B**: `config.VLM_MODELS` 에 항목 추가. 새 계열(family)이면 `verification.py` 에 어댑터 클래스 추가 후
  `load_vlm_verifier()` 분기에 등록

## 테스트 (GPU 불필요)

`source/` 에서 실행:

```bash
python tests/test_scoring.py            # B: 점수 집계
python tests/test_consensus.py          # A: 합의 로직 (FakeEmbedder)
python tests/test_extraction_parse.py   # A: JSON 파싱/구조 검증
python tests/test_verification_parse.py # B: 판정 파싱
```

각 테스트는 실제 모델 없이 mock 으로 로직만 검증하므로, 담당자는 상대방 모델/코드 없이도
자기 부분을 개발·검증할 수 있습니다.

## 사용 예 (코드)

```python
from safety_pipeline import config
from safety_pipeline.dataio import load_samples_from_data_dir, load_all_claim_sets
from safety_pipeline.stage_a import run_stage_a
from safety_pipeline.stage_b import run_stage_b
from safety_pipeline.analysis import summarize_model_comparison

samples = load_samples_from_data_dir(config.DEFAULT_DATA_DIR)
run_stage_a(samples)                                  # A: 저장까지
claim_sets = load_all_claim_sets(config.DEFAULT_STAGE_A_DIR)
summary_df, records = run_stage_b(claim_sets)         # B
summarize_model_comparison(summary_df)
```

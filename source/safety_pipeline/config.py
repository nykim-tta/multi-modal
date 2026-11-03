"""공유 설정 — 유형 체계, 유형 중요도, 비교 대상 모델, 파이프라인 파라미터.

이 파일의 CLAIM_TYPES / DEFAULT_TYPE_IMPORTANCE 는 A(추출·합의)와 B(검증·점수)
양쪽에 모두 영향을 줍니다. 변경 시 두 담당자가 함께 확인하세요.
"""

# ---------------------------------------------------------------------------
# 1) 유형 체계 (A·B 공통 계약)
# ---------------------------------------------------------------------------
CLAIM_TYPES = [
    "object_identity",  # 대상의 존재 여부, 종류, 정체        (예: SUV 한 대가 있다.)
    "attribute",        # 대상의 색상/상태/수치 등 속성        (예: SUV는 흰색이다.)
    "action",           # 대상의 행위/동작/상호작용            (예: SUV가 주차되어 있다.)
    "location",         # 대상/행위가 발생한 장소·위치         (예: SUV는 도로 왼쪽에 위치한다.)
    "temporal",         # 시간/시점 정보                       (예: 장면은 밤이다.)
    "text_content",     # 이미지 내 텍스트(표지판·번호판·문구) (예: 표지판에 '주정차 금지'라고 적혀 있다.)
    "context",          # 이미지만으로 검증 어려운 인과/추정   (예: SUV가 신호를 무시했다.)
]

VERDICT_SCORES = {"supported": 1.0, "contradicted": 0.0}

# 유형별 중요도(합 = 1.0). 최종 점수의 가중 평균에 사용됩니다.
# ※ 임의 확정하지 말고 사람 검증 세트에서 조정하는 것을 권장합니다.
# context 는 이미지로 검증이 어려워 대개 uncertain 처리되므로 낮게 둡니다.
DEFAULT_TYPE_IMPORTANCE = {
    "object_identity": 0.25,
    "attribute": 0.20,
    "action": 0.20,
    "location": 0.15,
    "temporal": 0.05,
    "text_content": 0.10,
    "context": 0.05,
}

# ---------------------------------------------------------------------------
# 2) 비교 대상 모델
#    A 담당: CLAIM_MODELS / EMBEDDING_MODELS 를 늘려 LLM·임베딩을 비교
#    B 담당: VLM_MODELS 를 늘려 VLM 을 비교
# ---------------------------------------------------------------------------
CLAIM_MODELS = {
    "qwen2.5-7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "dtype": "bfloat16",
        "device_map": "auto",
    },
    "kanana-1.5-8b": {
        "model_id": "kakaocorp/kanana-1.5-8b-instruct-2505",
        "dtype": "bfloat16",
        "device_map": "auto",
        # 카카오 Kanana 1.5 8B (llama 구조 → 네이티브 로드, remote code 불필요)
    },
    "gemma-2-9b-it": {
        "model_id": "google/gemma-2-9b-it",
        "dtype": "bfloat16",
        "device_map": "auto",
        # google/gemma-2 는 HF 라이선스 동의 + 토큰 로그인이 필요합니다.
    },
    # "llama-3.1-8b": {
    #     "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    #     "dtype": "bfloat16",
    #     "device_map": "auto",
    # },
}

VLM_MODELS = {
    "qwen2.5-vl-7b": {
        "family": "qwen2_5_vl",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "dtype": "bfloat16",
        "device_map": "auto",
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1024 * 28 * 28,
    },
}

EMBEDDING_MODELS = {
    "bge-m3": {
        "model_id": "BAAI/bge-m3",
        "device": "cpu",
    },
    # "multilingual-e5": {
    #     "model_id": "intfloat/multilingual-e5-large-instruct",
    #     "device": "cpu",
    # },
}

# ---------------------------------------------------------------------------
# 3) 파이프라인 파라미터
# ---------------------------------------------------------------------------
CLAIM_EXTRACTION_PARAMS = {
    "n_runs": 3,
    "temperature": 0.8,
    "top_p": 0.95,
    "base_seed": 42,
    "min_agreement": 0.6,
    "similarity_threshold": 0.84,
    "max_new_tokens": 1280,
}

# 최종 semantic_score 집계에서 제외할 유형.
# 제외된 유형도 type_scores / claim 상세에는 그대로 남아(수집됨) 확인할 수 있고,
# 오직 최종 점수 가중 평균에서만 빠집니다.
SCORING_EXCLUDED_TYPES = ["context"]

SCORING_PARAMS = {
    "type_importance": DEFAULT_TYPE_IMPORTANCE,
    "agreement_power": 1.0,
    "min_agreement_weight": 0.0,
    "uncertain_policy": "exclude",
    "excluded_types": SCORING_EXCLUDED_TYPES,
}

# 기존 노트북과의 호환을 위한 통합 딕셔너리
EXPERIMENT_CONFIG = {
    "claim_models": CLAIM_MODELS,
    "vlm_models": VLM_MODELS,
    "embedding_models": EMBEDDING_MODELS,
    "claim_extraction": CLAIM_EXTRACTION_PARAMS,
    "scoring": SCORING_PARAMS,
}

# ---------------------------------------------------------------------------
# 4) 기본 경로
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = "../data"
DEFAULT_STAGE_A_DIR = "./comparison_results/stage_a"   # A 산출물(claim 집합)
DEFAULT_STAGE_B_DIR = "./comparison_results/stage_b"   # B 산출물(검증·점수)

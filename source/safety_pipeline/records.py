"""A ↔ B 사이의 데이터 계약(seam).

두 단계가 주고받는 dict/데이터클래스 구조를 여기서 한 곳에 정의합니다.
A(추출·합의)와 B(검증·점수)는 이 파일의 구조만 신뢰하면 서로 독립적으로 작업할 수 있습니다.

--- claim dict 스키마 (A의 consensus 가 생산 → B가 소비) ---
{
    "claim_id": str,                 # "c1", "c2", ...
    "type": str,                     # config.CLAIM_TYPES 중 하나
    "text": str,                     # claim 문장
    "subject": str,                  # claim 대상
    "source_span": str,              # 원문 근거 부분 (optional)
    "value": str,                    # 속성/위치/방향 등의 값 (유형에 따라 optional)
    "agreement": float,              # votes / total_runs
    "votes": int,
    "total_runs": int,
    "mean_semantic_similarity": float,
    "variants": list[str],
    "source_spans": list[str],
}

--- verdict dict 스키마 (B의 verification 이 생산) ---
{
    "claim_id": str,
    "verdict": "supported" | "contradicted" | "uncertain",
    "support_score": float | None,   # uncertain 이면 None
    "evidence": str,
}
"""

from dataclasses import dataclass, field, asdict


@dataclass
class ClaimSet:
    """Stage A 산출물. 한 (샘플 × LLM × 임베딩) 조합의 최종 claim 집합."""
    sample_id: str
    image_path: str
    description: str
    claim_model: str
    embedding_model: str
    claims: list = field(default_factory=list)      # list[dict] (claim dict 스키마)
    raw_runs: list = field(default_factory=list)     # list[dict] (n_runs개의 원시 추출 결과)
    extraction_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ClaimSet":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in allowed})


@dataclass
class EvaluationRecord:
    """Stage B 산출물. ClaimSet + VLM 판정 + 점수."""
    sample_id: str
    image_path: str
    description: str
    claim_model: str
    vlm_model: str
    embedding_model: str
    claims: list = field(default_factory=list)
    verdicts: list = field(default_factory=list)
    scoring: dict = field(default_factory=dict)
    claim_extraction_seconds: float = 0.0
    verification_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvaluationRecord":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in allowed})

"""[담당 B] 점수 집계 — claim별 → 유형별 → 최종 의미 정확성 점수.

claim 가중치 w_i = agreement_i ** agreement_power
유형 점수 S_k = Σ(w_i · support_score_i) / Σ w_i   (uncertain 제외)
최종 점수  = Σ(importance_k · S_k) / Σ importance_k   (관측된 유형만)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import CLAIM_TYPES


@dataclass
class ScoringResult:
    semantic_score: Optional[float]
    type_scores: dict
    type_details: dict
    evaluable_claim_rate: float
    mean_claim_agreement: float
    claim_count: int
    evaluable_claim_count: int
    supported_count: int
    contradicted_count: int
    uncertain_count: int


def score_claims(
    claims: list,
    verdicts: list,
    type_importance: dict,
    agreement_power: float = 1.0,
    min_agreement_weight: float = 0.0,
    excluded_types=None,
) -> ScoringResult:
    excluded = set(excluded_types or [])
    verdict_map = {v["claim_id"]: v for v in verdicts}
    type_scores, type_details = {}, {}
    evaluable_count = supported_count = contradicted_count = uncertain_count = 0

    for claim_type in CLAIM_TYPES:
        numerator = denominator = 0.0
        rows = []
        for claim in [c for c in claims if c["type"] == claim_type]:
            verdict = verdict_map.get(claim["claim_id"], {"verdict": "uncertain", "support_score": None})
            agreement = float(claim.get("agreement", 1.0))
            weight = max(min_agreement_weight, agreement ** agreement_power)
            score = verdict.get("support_score")
            if verdict["verdict"] == "uncertain" or score is None:
                uncertain_count += 1
                included = False
            else:
                evaluable_count += 1
                supported_count += int(verdict["verdict"] == "supported")
                contradicted_count += int(verdict["verdict"] == "contradicted")
                numerator += weight * float(score)
                denominator += weight
                included = True
            rows.append({
                "claim_id": claim["claim_id"], "text": claim["text"],
                "agreement": agreement, "claim_weight": weight,
                "verdict": verdict["verdict"], "support_score": score,
                "included": included,
            })
        type_scores[claim_type] = round(numerator / denominator, 4) if denominator > 0 else None
        type_details[claim_type] = rows

    final_num = final_den = 0.0
    for claim_type, type_score in type_scores.items():
        if type_score is None:
            continue
        if claim_type in excluded:          # 수집은 하되 최종 점수에서는 배제 (예: context)
            continue
        importance = float(type_importance.get(claim_type, 0.0))
        final_num += importance * type_score
        final_den += importance

    semantic_score = round(final_num / final_den, 4) if final_den > 0 else None
    total_claims = len(claims)
    return ScoringResult(
        semantic_score=semantic_score,
        type_scores=type_scores,
        type_details=type_details,
        evaluable_claim_rate=round(evaluable_count / total_claims, 4) if total_claims else 0.0,
        mean_claim_agreement=round(float(np.mean([c.get("agreement", 1.0) for c in claims])), 4) if claims else 0.0,
        claim_count=total_claims,
        evaluable_claim_count=evaluable_count,
        supported_count=supported_count,
        contradicted_count=contradicted_count,
        uncertain_count=uncertain_count,
    )

"""[담당 B] 점수 집계 테스트 — VLM 없이 score_claims 로직 검증.

실행: source/ 에서  python tests/test_scoring.py   (또는 pytest)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety_pipeline.config import DEFAULT_TYPE_IMPORTANCE
from safety_pipeline.scoring import score_claims
from tests.fixtures import make_claims_and_verdicts


def test_score_basic():
    claims, verdicts = make_claims_and_verdicts()
    r = score_claims(claims, verdicts, DEFAULT_TYPE_IMPORTANCE)

    # uncertain(c2) 은 평가에서 제외
    assert r.uncertain_count == 1
    assert r.evaluable_claim_count == 3
    assert r.supported_count == 2
    assert r.contradicted_count == 1
    assert r.claim_count == 4
    assert abs(r.evaluable_claim_rate - 0.75) < 1e-6

    # attribute 는 c2 하나뿐인데 uncertain → 점수 None
    assert r.type_scores["attribute"] is None
    assert r.type_scores["object_identity"] == 0.9
    assert r.type_scores["location"] == 0.8
    assert r.type_scores["action"] == 0.1
    assert r.type_scores["context"] is None

    # 최종 점수: 관측된 유형(object_identity, location, action)만 중요도 가중 평균
    imp = DEFAULT_TYPE_IMPORTANCE
    num = imp["object_identity"] * 0.9 + imp["location"] * 0.8 + imp["action"] * 0.1
    den = imp["object_identity"] + imp["location"] + imp["action"]
    assert abs(r.semantic_score - round(num / den, 4)) < 1e-4
    print("[PASS] test_score_basic  semantic_score =", r.semantic_score)


def test_context_excluded_from_score():
    # context claim 은 type_scores 에는 남지만(수집), 최종 semantic_score 에는 영향 없어야 함
    claims = [
        {"claim_id": "c1", "type": "object_identity", "text": "SUV가 있다.", "agreement": 1.0},
        {"claim_id": "c2", "type": "context", "text": "SUV가 신호를 무시했다.", "agreement": 1.0},
    ]
    verdicts = [
        {"claim_id": "c1", "verdict": "supported", "support_score": 0.8, "evidence": "보임"},
        {"claim_id": "c2", "verdict": "supported", "support_score": 0.2, "evidence": "추정"},
    ]
    r = score_claims(claims, verdicts, DEFAULT_TYPE_IMPORTANCE, excluded_types=["context"])
    # context 는 계산·보관됨
    assert r.type_scores["context"] == 0.2
    # 하지만 최종 점수는 object_identity 만으로 결정 (context 배제)
    assert r.semantic_score == 0.8
    print("[PASS] test_context_excluded_from_score  semantic_score =", r.semantic_score,
          "| context type_score =", r.type_scores["context"])


def test_empty_claims():
    r = score_claims([], [], DEFAULT_TYPE_IMPORTANCE)
    assert r.semantic_score is None
    assert r.claim_count == 0
    assert r.evaluable_claim_rate == 0.0
    print("[PASS] test_empty_claims")


if __name__ == "__main__":
    test_score_basic()
    test_context_excluded_from_score()
    test_empty_claims()
    print("\n[scoring] 모든 테스트 통과")

"""[담당 B] 검증 출력 파싱 테스트 — validate_verdicts (VLM 없이).

실행: source/ 에서  python tests/test_verification_parse.py   (또는 pytest)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety_pipeline.verification import validate_verdicts


def test_validate_verdicts_basic():
    claims = [
        {"claim_id": "c1", "type": "object_identity", "text": "소화전이 있다."},
        {"claim_id": "c2", "type": "attribute", "text": "파손됨."},
        {"claim_id": "c3", "type": "location", "text": "보도에 있다."},
    ]
    # c3 은 모델이 누락 → 기본 uncertain 처리되어야 함
    raw = [
        {"claim_id": "c1", "verdict": "supported", "support_score": 0.9, "evidence": "보임"},
        {"claim_id": "c2", "verdict": "contradicted", "evidence": "멀쩡함"},  # score 누락 → 0.0
    ]
    out = validate_verdicts(raw, claims)
    by = {v["claim_id"]: v for v in out}

    assert len(out) == 3
    assert by["c1"]["verdict"] == "supported" and by["c1"]["support_score"] == 0.9
    assert by["c2"]["verdict"] == "contradicted" and by["c2"]["support_score"] == 0.0
    assert by["c3"]["verdict"] == "uncertain" and by["c3"]["support_score"] is None
    print("[PASS] test_validate_verdicts_basic")


def test_validate_verdicts_bad_label_and_range():
    claims = [{"claim_id": "c1", "type": "attribute", "text": "x"}]
    raw = [{"claim_id": "c1", "verdict": "maybe", "support_score": 5}]  # 잘못된 라벨 + 범위 초과
    out = validate_verdicts(raw, claims)
    assert out[0]["verdict"] == "uncertain"        # 잘못된 라벨 → uncertain
    # 값이 주어졌으므로 0~1 로 클램프(5 → 1.0). (uncertain 이라 점수는 채점에서 어차피 제외됨)
    assert out[0]["support_score"] == 1.0
    print("[PASS] test_validate_verdicts_bad_label_and_range")


def test_validate_verdicts_not_list():
    try:
        validate_verdicts({"claim_id": "c1"}, [{"claim_id": "c1", "type": "attribute", "text": "x"}])
        assert False, "리스트가 아니면 ValueError 여야 함"
    except ValueError:
        print("[PASS] test_validate_verdicts_not_list")


if __name__ == "__main__":
    test_validate_verdicts_basic()
    test_validate_verdicts_bad_label_and_range()
    test_validate_verdicts_not_list()
    print("\n[verification parse] 모든 테스트 통과")

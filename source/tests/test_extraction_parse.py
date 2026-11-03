"""[담당 A] 파싱/검증 테스트 — extract_json 복구 + validate_claim_structure.

실행: source/ 에서  python tests/test_extraction_parse.py   (또는 pytest)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety_pipeline.common import extract_json
from safety_pipeline.extraction import validate_claim_structure


def test_extract_json_trailing_comma():
    bad = '{"object_identity": [{"text": "소화전이 있다."}], "attribute": [],}'
    out = extract_json(bad)  # 후행 쉼표 → 복구되어야 함
    assert out["object_identity"][0]["text"] == "소화전이 있다."
    print("[PASS] test_extract_json_trailing_comma")


def test_extract_json_codefence():
    bad = '```json\n{"context": []}\n```'
    out = extract_json(bad)
    assert out == {"context": []}
    print("[PASS] test_extract_json_codefence")


def test_validate_structure():
    raw = {
        "object_identity": [{"text": "소화전이 있다.", "subject": "소화전"}],
        "attribute": [{"text": "  "}],          # 빈 텍스트 → 제거
        "location": "not a list",                # 잘못된 형 → 빈 배열
        "text_content": [{"text": "STOP", "value": "STOP"}],
    }
    out = validate_claim_structure(raw)
    # 7개 유형 키가 항상 존재
    assert set(out.keys()) == {
        "object_identity", "attribute", "action", "location",
        "temporal", "text_content", "context",
    }
    assert len(out["object_identity"]) == 1
    assert out["object_identity"][0]["type"] == "object_identity"
    assert out["attribute"] == []          # 빈 텍스트 제거됨
    assert out["location"] == []           # 잘못된 형 → 빈 배열
    assert out["text_content"][0]["text"] == "STOP"
    print("[PASS] test_validate_structure")


if __name__ == "__main__":
    test_extract_json_trailing_comma()
    test_extract_json_codefence()
    test_validate_structure()
    print("\n[extraction parse] 모든 테스트 통과")

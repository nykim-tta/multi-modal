"""[담당 A] 합의 로직 테스트 — 실제 임베딩 모델 없이 FakeEmbedder 로 검증.

실행: source/ 에서  python tests/test_consensus.py   (또는 pytest)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety_pipeline.consensus import semantic_consensus, claims_compatible
from tests.fixtures import FakeEmbedder, make_run_structures


def test_agreement_counts():
    runs = make_run_structures()  # ident 3회, damaged 2회
    claims = semantic_consensus(
        runs, FakeEmbedder(), similarity_threshold=0.84, min_agreement=0.6, verbose=False
    )
    by_type = {c["type"]: c for c in claims}

    # 두 claim 모두 min_agreement(0.6) 이상이라 채택됨
    assert "object_identity" in by_type
    assert "attribute" in by_type
    assert by_type["object_identity"]["agreement"] == 1.0      # 3/3
    assert by_type["object_identity"]["votes"] == 3
    assert abs(by_type["attribute"]["agreement"] - 0.6667) < 1e-3  # 2/3
    assert by_type["attribute"]["votes"] == 2
    print("[PASS] test_agreement_counts  agreements =",
          {t: c["agreement"] for t, c in by_type.items()})


def test_min_agreement_filter():
    runs = make_run_structures()
    # min_agreement 를 0.7 로 올리면 2/3(=0.667)짜리 attribute 는 기각
    claims = semantic_consensus(
        runs, FakeEmbedder(), similarity_threshold=0.84, min_agreement=0.7, verbose=False
    )
    types = {c["type"] for c in claims}
    assert "object_identity" in types
    assert "attribute" not in types
    print("[PASS] test_min_agreement_filter  남은 유형 =", types)


def test_compatibility():
    a = {"type": "object_identity", "subject": "소화전"}
    b = {"type": "object_identity", "subject": "표지판"}
    c = {"type": "attribute", "subject": "소화전"}
    assert claims_compatible(a, dict(a)) is True
    assert claims_compatible(a, b) is False   # subject 다름
    assert claims_compatible(a, c) is False   # type 다름
    print("[PASS] test_compatibility")


if __name__ == "__main__":
    test_agreement_counts()
    test_min_agreement_filter()
    test_compatibility()
    print("\n[consensus] 모든 테스트 통과")

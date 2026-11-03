"""[공유] gold 정합성 지표 테스트 — 실제 임베딩 모델 없이 FakeEmbedder 로 검증.

실행: source/ 에서  python tests/test_gold_eval.py   (또는 pytest)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety_pipeline.gold_eval import similarity_metrics, evaluate_overall
from safety_pipeline.records import ClaimSet
from tests.fixtures import FakeEmbedder


def test_identical_sets():
    texts = ["소화전이 있다.", "소화전이 파손되어 있다."]
    m = similarity_metrics(texts, list(texts), FakeEmbedder(), threshold=0.7)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
    assert abs(m["soft_f1"] - 1.0) < 1e-6
    print("[PASS] test_identical_sets")


def test_empty_cases():
    fe = FakeEmbedder()
    both = similarity_metrics([], [], fe)
    assert both["f1"] == 1.0 and both["soft_f1"] == 1.0        # 둘 다 없음 → 완전 일치
    one = similarity_metrics(["소화전이 있다."], [], fe)
    assert one["f1"] == 0.0 and one["soft_f1"] == 0.0          # 한쪽만 비면 0
    print("[PASS] test_empty_cases")


def test_partial_overlap():
    # 예측 2개, gold 2개 중 1개("소화전이 있다.")만 정확히 공유
    pred = ["소화전이 있다.", "표지판이 파랗다."]
    gold = ["소화전이 있다.", "도로가 넓다."]
    m = similarity_metrics(pred, gold, FakeEmbedder(), threshold=0.7)
    assert m["precision"] == 0.5   # 예측 2개 중 1개만 gold와 매칭
    assert m["recall"] == 0.5      # gold 2개 중 1개만 예측과 매칭
    assert m["f1"] == 0.5
    print("[PASS] test_partial_overlap  ->", {k: m[k] for k in ("precision", "recall", "f1")})


def test_evaluate_overall():
    cs = ClaimSet(
        sample_id="construction__idx_00001_0",
        image_path="x.jpg", description="...",
        claim_model="m", embedding_model="e",
        claims=[
            {"claim_id": "c1", "type": "object_identity", "text": "소화전이 있다."},
            {"claim_id": "c2", "type": "attribute", "text": "소화전은 빨갛다."},
        ],
    )
    gold_map = {
        "construction__idx_00001_0": {
            "gold_claims": {
                "object_identity": ["소화전이 있다."],
                "attribute": ["소화전은 빨갛다."],
                "action": [], "location": [], "temporal": [], "text_content": [], "context": [],
            }
        }
    }
    rows = evaluate_overall([cs], gold_map, FakeEmbedder(), threshold=0.7)
    assert len(rows) == 1
    r = rows[0]
    assert r["category"] == "construction"
    assert r["n_pred"] == 2 and r["n_gold"] == 2
    assert r["f1"] == 1.0
    print("[PASS] test_evaluate_overall  ->", r["sample_id"], "f1 =", r["f1"])


if __name__ == "__main__":
    test_identical_sets()
    test_empty_cases()
    test_partial_overlap()
    test_evaluate_overall()
    print("\n[gold_eval] 모든 테스트 통과")

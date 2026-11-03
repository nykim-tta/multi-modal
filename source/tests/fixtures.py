"""테스트용 mock 데이터/객체 — 실제 모델(LLM/VLM/임베딩) 없이 로직을 검증한다."""

import hashlib
import numpy as np


class FakeEmbedder:
    """텍스트별로 결정적(deterministic) 단위 벡터를 반환하는 가짜 임베더.

    동일 텍스트 → 동일 벡터(유사도 1.0), 다른 텍스트 → 거의 직교.
    실제 bge-m3 없이 consensus 클러스터링/합의 로직을 테스트할 수 있다.
    """
    name = "fake-embedder"
    dim = 64

    def encode(self, texts: list) -> np.ndarray:
        vecs = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            # 해시 바이트를 float 벡터로 확장
            raw = np.frombuffer((h * ((self.dim // len(h)) + 1))[: self.dim], dtype=np.uint8).astype(np.float64)
            v = raw - raw.mean()
            n = np.linalg.norm(v)
            vecs.append(v / n if n > 0 else v)
        return np.array(vecs)


def make_run_structures():
    """3회 run 결과 mock. 'object_identity: 소화전' 은 3회 모두, 'attribute: 파손' 은 2회만 등장."""
    ident = {"text": "소화전이 있다.", "source_span": "소화전", "subject": "소화전"}
    damaged = {"text": "소화전이 파손되어 있다.", "source_span": "파손", "subject": "소화전", "value": "파손"}
    empty = {"object_identity": [], "attribute": [], "action": [], "location": [],
             "temporal": [], "text_content": [], "context": []}

    run0 = {**empty, "object_identity": [dict(ident)], "attribute": [dict(damaged)]}
    run1 = {**empty, "object_identity": [dict(ident)], "attribute": [dict(damaged)]}
    run2 = {**empty, "object_identity": [dict(ident)]}  # attribute 누락 → agreement 2/3
    return [run0, run1, run2]


def make_claims_and_verdicts():
    """score_claims 테스트용 claim/verdict mock (여러 유형 + uncertain/contradicted 포함)."""
    claims = [
        {"claim_id": "c1", "type": "object_identity", "text": "소화전이 있다.", "agreement": 1.0},
        {"claim_id": "c2", "type": "attribute", "text": "소화전이 파손되어 있다.", "agreement": 0.6667},
        {"claim_id": "c3", "type": "location", "text": "소화전이 보도에 있다.", "agreement": 1.0},
        {"claim_id": "c4", "type": "action", "text": "소화전이 넘어져 있다.", "agreement": 1.0},
    ]
    verdicts = [
        {"claim_id": "c1", "verdict": "supported", "support_score": 0.9, "evidence": "보임"},
        {"claim_id": "c2", "verdict": "uncertain", "support_score": None, "evidence": "판단 불가"},
        {"claim_id": "c3", "verdict": "supported", "support_score": 0.8, "evidence": "보도 위"},
        {"claim_id": "c4", "verdict": "contradicted", "support_score": 0.1, "evidence": "서 있음"},
    ]
    return claims, verdicts

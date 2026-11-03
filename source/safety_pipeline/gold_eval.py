"""[공유] Gold standard 정합성 평가 — 추출 claim vs gold_claims 를 임베딩 유사도로 비교.

핵심 아이디어(soft/hard P·R·F1):
- 예측 claim E, gold claim G 를 임베딩(코사인 유사도)으로 대조.
- 각 예측 claim의 "gold 중 최고 유사도" 평균 = soft precision, 임계값 이상 비율 = hard precision.
- 각 gold claim의 "예측 중 최고 유사도" 평균 = soft recall, 임계값 이상 비율 = hard recall.
- F1 = precision·recall 조화평균.

embedder 는 encode(list[str]) -> (n, d) 정규화 벡터를 반환하는 객체면 무엇이든 됩니다
(예: consensus.ClaimEmbedder). 서로 다른 임베딩 모델을 넣어 비교합니다.
"""

import json
from pathlib import Path

import numpy as np

from .config import CLAIM_TYPES


def load_gold_by_sample(data_dir) -> dict:
    """data2 의 *_gold.json 을 읽어 {sample_id: {"gold_claims":{...}, "description":...}} 반환."""
    out = {}
    for p in sorted(Path(data_dir).rglob("*_gold.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        index = d.get("index")
        gold = d.get("gold_claims", {})
        for rep in d.get("image_info", {}).get("image_report", []):
            sid = f"{index}_{rep.get('image_index')}"
            out[sid] = {"gold_claims": gold, "description": rep.get("description")}
    return out


def claims_to_texts_by_type(claims: list) -> dict:
    """추출된 consensus claim(list[dict]) -> {type: [text, ...]}."""
    by_type = {t: [] for t in CLAIM_TYPES}
    for c in claims:
        t = c.get("type")
        txt = str(c.get("text", "")).strip()
        if t in by_type and txt:
            by_type[t].append(txt)
    return by_type


def gold_to_texts_by_type(gold_claims: dict) -> dict:
    """gold_claims dict -> {type: [text, ...]} (누락 유형은 빈 리스트)."""
    by_type = {t: [] for t in CLAIM_TYPES}
    for t in CLAIM_TYPES:
        for txt in gold_claims.get(t, []) or []:
            txt = str(txt).strip()
            if txt:
                by_type[t].append(txt)
    return by_type


def _flatten(by_type: dict, types=None) -> list:
    types = types or CLAIM_TYPES
    return [txt for t in types for txt in by_type.get(t, [])]


def similarity_metrics(pred_texts: list, gold_texts: list, embedder, threshold: float = 0.7) -> dict:
    """예측/gold claim 텍스트를 임베딩 유사도로 대조해 정합성 지표를 계산."""
    n_p, n_g = len(pred_texts), len(gold_texts)
    empty = dict(soft_precision=None, soft_recall=None, soft_f1=None,
                 precision=None, recall=None, f1=None,
                 mean_pred_best=None, mean_gold_best=None, n_pred=n_p, n_gold=n_g)
    if n_p == 0 and n_g == 0:
        # 둘 다 없음 → 완전 일치로 간주
        return dict(soft_precision=1.0, soft_recall=1.0, soft_f1=1.0,
                    precision=1.0, recall=1.0, f1=1.0,
                    mean_pred_best=None, mean_gold_best=None, n_pred=0, n_gold=0)
    if n_p == 0 or n_g == 0:
        # 한쪽만 비어 있음 → 0점
        return {**empty, "soft_precision": 0.0, "soft_recall": 0.0, "soft_f1": 0.0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0}

    pe = np.asarray(embedder.encode(list(pred_texts)))
    ge = np.asarray(embedder.encode(list(gold_texts)))
    sim = pe @ ge.T                       # 정규화 임베딩 → 코사인 유사도
    pred_best = sim.max(axis=1)           # 각 예측 claim의 최고 gold 유사도
    gold_best = sim.max(axis=0)           # 각 gold claim의 최고 예측 유사도

    sp, sr = float(pred_best.mean()), float(gold_best.mean())
    sf = 2 * sp * sr / (sp + sr) if (sp + sr) > 0 else 0.0
    hp, hr = float((pred_best >= threshold).mean()), float((gold_best >= threshold).mean())
    hf = 2 * hp * hr / (hp + hr) if (hp + hr) > 0 else 0.0
    return dict(
        soft_precision=round(sp, 4), soft_recall=round(sr, 4), soft_f1=round(sf, 4),
        precision=round(hp, 4), recall=round(hr, 4), f1=round(hf, 4),
        mean_pred_best=round(sp, 4), mean_gold_best=round(sr, 4),
        n_pred=n_p, n_gold=n_g,
    )


def _category_of(sample_id: str) -> str:
    return sample_id.split("__")[0] if "__" in sample_id else ""


def evaluate_overall(claim_sets: list, gold_map: dict, embedder, threshold: float = 0.7) -> list:
    """샘플별 전체(유형 무시) 정합성 지표 리스트."""
    rows = []
    for cs in claim_sets:
        gold = gold_map.get(cs.sample_id, {}).get("gold_claims", {})
        pred = _flatten(claims_to_texts_by_type(cs.claims))
        goldt = _flatten(gold_to_texts_by_type(gold))
        m = similarity_metrics(pred, goldt, embedder, threshold)
        rows.append({"sample_id": cs.sample_id, "category": _category_of(cs.sample_id), **m})
    return rows


def match_detail_table(claim_sets: list, gold_map: dict, embedder, threshold: float = 0.7) -> list:
    """claim 단위 상세 매칭 표.

    각 추출(pred) claim → 가장 유사한 gold claim + 유사도, 각 gold claim → 가장 유사한 추출 claim + 유사도.
    "어떤 claim이 뽑혔는지"와 "그게 gold와 얼마나 맞는지"를 한눈에 볼 수 있습니다.
    반환 각 행: {sample_id, side(pred/gold), claim_type, text, best_match, best_sim, matched}
    """
    rows = []
    for cs in claim_sets:
        gold = gold_map.get(cs.sample_id, {}).get("gold_claims", {})
        pred_bt = claims_to_texts_by_type(cs.claims)
        gold_bt = gold_to_texts_by_type(gold)
        pred = [(t, txt) for t in CLAIM_TYPES for txt in pred_bt[t]]
        goldl = [(t, txt) for t in CLAIM_TYPES for txt in gold_bt[t]]

        pe = np.asarray(embedder.encode([x[1] for x in pred])) if pred else None
        ge = np.asarray(embedder.encode([x[1] for x in goldl])) if goldl else None

        for i, (t, txt) in enumerate(pred):
            if ge is not None:
                sims = pe[i] @ ge.T
                j = int(sims.argmax()); best = float(sims[j]); bmatch = goldl[j][1]
            else:
                best, bmatch = 0.0, None
            rows.append({"sample_id": cs.sample_id, "side": "pred", "claim_type": t, "text": txt,
                         "best_match": bmatch, "best_sim": round(best, 4), "matched": best >= threshold})

        for j, (t, txt) in enumerate(goldl):
            if pe is not None:
                sims = ge[j] @ pe.T
                i = int(sims.argmax()); best = float(sims[i]); bmatch = pred[i][1]
            else:
                best, bmatch = 0.0, None
            rows.append({"sample_id": cs.sample_id, "side": "gold", "claim_type": t, "text": txt,
                         "best_match": bmatch, "best_sim": round(best, 4), "matched": best >= threshold})
    return rows


def evaluate_by_type(claim_sets: list, gold_map: dict, embedder, threshold: float = 0.7) -> list:
    """샘플×유형 단위 정합성 지표 리스트 (예측·gold 둘 다 비어있는 칸은 제외)."""
    rows = []
    for cs in claim_sets:
        gold = gold_map.get(cs.sample_id, {}).get("gold_claims", {})
        pred_bt = claims_to_texts_by_type(cs.claims)
        gold_bt = gold_to_texts_by_type(gold)
        for t in CLAIM_TYPES:
            p, g = pred_bt[t], gold_bt[t]
            if not p and not g:
                continue  # 둘 다 없는 유형은 집계에서 제외
            m = similarity_metrics(p, g, embedder, threshold)
            rows.append({"sample_id": cs.sample_id, "claim_type": t, **m})
    return rows

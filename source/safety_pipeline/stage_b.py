"""[담당 B] Stage B 오케스트레이션 — A의 claim 집합을 VLM별로 검증·채점.

입력: A가 만든 ClaimSet 목록 (메모리로 직접 전달하거나 load_claim_sets 로 파일에서 로드)
산출물: 각 (claim_model × vlm_model) 조합의 EvaluationRecord + 요약 테이블
  stageB_details__<claim_model>__<vlm_model>.json
  stageB_summary.csv / .xlsx
"""

import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import EXPERIMENT_CONFIG, DEFAULT_STAGE_B_DIR, CLAIM_TYPES
from .records import EvaluationRecord
from .verification import load_vlm_verifier
from .scoring import score_claims
from .dataio import save_json


def flatten_record(record: EvaluationRecord) -> dict:
    row = {
        "sample_id": record.sample_id,
        "image_path": record.image_path,
        "description": record.description,
        "claim_model": record.claim_model,
        "vlm_model": record.vlm_model,
        "embedding_model": record.embedding_model,
        "semantic_score": record.scoring["semantic_score"],
        "evaluable_claim_rate": record.scoring["evaluable_claim_rate"],
        "mean_claim_agreement": record.scoring["mean_claim_agreement"],
        "claim_count": record.scoring["claim_count"],
        "supported_count": record.scoring["supported_count"],
        "contradicted_count": record.scoring["contradicted_count"],
        "uncertain_count": record.scoring["uncertain_count"],
        "claim_extraction_seconds": record.claim_extraction_seconds,
        "verification_seconds": record.verification_seconds,
    }
    for t in CLAIM_TYPES:
        row[f"score_{t}"] = record.scoring["type_scores"].get(t)
    return row


def claims_by_sample(records: list) -> list:
    output = []
    for record in records:
        verdict_map = {v["claim_id"]: v for v in record.verdicts}
        claims = []
        for claim in record.claims:
            verdict = verdict_map.get(claim["claim_id"], {"verdict": "uncertain", "evidence": ""})
            claims.append({
                "claim_id": claim["claim_id"],
                "claim_type": claim["type"],
                "claim_text": claim["text"],
                "verdict": verdict["verdict"],
                "evidence": verdict.get("evidence", ""),
            })
        output.append({
            "sample_id": record.sample_id,
            "image_path": record.image_path,
            "description": record.description,
            "claims": claims,
        })
    return output


def run_stage_b(
    claim_sets_by_combo: dict,
    experiment_config: dict = EXPERIMENT_CONFIG,
    output_dir: str = DEFAULT_STAGE_B_DIR,
    save: bool = True,
    verbose: bool = True,
):
    """A의 claim 집합을 모든 vlm_model 로 검증·채점한다.

    claim_sets_by_combo: {(claim_model, embedding_model): list[ClaimSet]}
        (stage_a.run_stage_a 의 반환값, 또는 load_claim_sets 로 구성)
    반환: (summary_df, all_records)
    """
    output_path = Path(output_dir)
    if save:
        output_path.mkdir(parents=True, exist_ok=True)

    scoring_cfg = experiment_config["scoring"]
    all_records = []

    for vlm_name, vlm_cfg in experiment_config["vlm_models"].items():
        if verbose:
            print(f"\n########## Stage B: VLM={vlm_name} ##########")
        verifier = load_vlm_verifier(vlm_name, vlm_cfg)

        for (claim_name, emb_name), claim_sets in claim_sets_by_combo.items():
            combo_records = []
            for cs in claim_sets:
                start = time.perf_counter()
                verdicts = verifier.verify(cs.image_path, cs.claims)
                verify_seconds = time.perf_counter() - start
                scoring = score_claims(
                    cs.claims, verdicts,
                    type_importance=scoring_cfg["type_importance"],
                    agreement_power=scoring_cfg["agreement_power"],
                    min_agreement_weight=scoring_cfg["min_agreement_weight"],
                    excluded_types=scoring_cfg.get("excluded_types"),
                )
                rec = EvaluationRecord(
                    sample_id=cs.sample_id,
                    image_path=cs.image_path,
                    description=cs.description,
                    claim_model=claim_name,
                    vlm_model=vlm_name,
                    embedding_model=emb_name,
                    claims=cs.claims,
                    verdicts=verdicts,
                    scoring=asdict(scoring),
                    claim_extraction_seconds=cs.extraction_seconds,
                    verification_seconds=round(verify_seconds, 3),
                )
                combo_records.append(rec)
                all_records.append(rec)

            if save:
                save_json([asdict(r) for r in combo_records],
                          output_path / f"stageB_details__{claim_name}__{vlm_name}.json")
                save_json(claims_by_sample(combo_records),
                          output_path / f"stageB_claims__{claim_name}__{vlm_name}.json")

        verifier.unload()

    summary_df = pd.DataFrame([flatten_record(r) for r in all_records])
    if save and not summary_df.empty:
        summary_df.to_csv(output_path / "stageB_summary.csv", index=False, encoding="utf-8-sig")
        try:
            summary_df.to_excel(output_path / "stageB_summary.xlsx", index=False)
        except Exception as e:
            print(f"[알림] xlsx 저장 건너뜀: {e}")
    return summary_df, all_records

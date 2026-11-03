"""[담당 A] Stage A 오케스트레이션 — LLM(×임베딩)별로 claim 집합을 추출·저장.

산출물: 각 (claim_model × embedding_model) 조합마다
  stageA_claims__<claim_model>__<embedding_model>.json   (list[ClaimSet])
→ 이것이 B단계의 입력이 됩니다. A는 이 파일들만 비교하면 LLM 추출 품질을 볼 수 있습니다.
"""

import time
from pathlib import Path

from .config import EXPERIMENT_CONFIG, DEFAULT_STAGE_A_DIR
from .common import stable_id
from .records import ClaimSet
from .extraction import TransformersClaimExtractor
from .consensus import ClaimEmbedder, extract_consensus_claims
from .dataio import save_claim_sets, stage_a_filename, save_json


def _sample_id(sample: dict) -> str:
    return str(sample.get("sample_id") or stable_id(sample["image_path"], sample["description"]))


def run_stage_a(
    samples: list,
    experiment_config: dict = EXPERIMENT_CONFIG,
    output_dir: str = DEFAULT_STAGE_A_DIR,
    save: bool = True,
    verbose: bool = True,
) -> dict:
    """모든 claim_model × embedding_model 조합에 대해 claim 집합을 생성한다.

    반환: {(claim_model, embedding_model): list[ClaimSet]}
    """
    output_path = Path(output_dir)
    if save:
        output_path.mkdir(parents=True, exist_ok=True)

    results = {}
    extraction_cfg = experiment_config["claim_extraction"]

    for emb_name, emb_cfg in experiment_config["embedding_models"].items():
        embedder = ClaimEmbedder(emb_name, emb_cfg)

        for claim_name, claim_cfg in experiment_config["claim_models"].items():
            if verbose:
                print(f"\n########## Stage A: LLM={claim_name}, EMB={emb_name} ##########")
            extractor = TransformersClaimExtractor(claim_name, claim_cfg)
            claim_sets = []

            for sample in samples:
                sid = _sample_id(sample)
                if verbose:
                    print(f"\n----- 샘플 {sid} -----")
                claims, raw_runs, seconds = extract_consensus_claims(
                    extractor, embedder, sample["description"],
                    extraction_cfg, return_runs=True, verbose=verbose,
                )
                claim_sets.append(ClaimSet(
                    sample_id=sid,
                    image_path=sample["image_path"],
                    description=sample["description"],
                    claim_model=claim_name,
                    embedding_model=emb_name,
                    claims=claims,
                    raw_runs=raw_runs,
                    extraction_seconds=round(seconds, 3),
                ))

            extractor.unload()
            results[(claim_name, emb_name)] = claim_sets

            if save:
                out = output_path / stage_a_filename(claim_name, emb_name)
                save_claim_sets(claim_sets, out)
                # 사람이 보기 쉬운 요약(원시 run 제외)도 함께 저장
                readable = [{
                    "sample_id": cs.sample_id,
                    "image_path": cs.image_path,
                    "description": cs.description,
                    "claims": [
                        {"claim_id": c["claim_id"], "type": c["type"], "text": c["text"],
                         "agreement": c.get("agreement"), "variants": c.get("variants")}
                        for c in cs.claims
                    ],
                } for cs in claim_sets]
                save_json(readable, output_path / f"readable__{claim_name}__{emb_name}.json")
                if verbose:
                    print(f"\n[저장] {out}")

        embedder.unload()

    return results

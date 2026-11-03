"""[담당 A] 의미 기반 합의 — 임베딩 어댑터 + 유형 내 claim 군집화/합의도 계산.

extract_consensus_claims(extractor, embedder, description, params) 가 A단계의 핵심 진입점.
extractor 는 extract_runs() 를 제공하는 객체면 무엇이든 됩니다(덕 타이핑).
"""

import re
import time
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from .config import CLAIM_TYPES
from .common import clear_memory


class ClaimEmbedder:
    def __init__(self, name: str, config: dict):
        self.name = name
        self.model = SentenceTransformer(config["model_id"], device=config.get("device", "cpu"))

    def encode(self, texts: list) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def unload(self):
        del self.model
        clear_memory()


def _norm_subject(x: Any) -> str:
    return re.sub(r"\s+", "", str(x or "").lower())


def claims_compatible(a: dict, b: dict) -> bool:
    if a["type"] != b["type"]:
        return False
    sa, sb = _norm_subject(a.get("subject")), _norm_subject(b.get("subject"))
    if sa and sb and sa != sb and sa not in sb and sb not in sa:
        return False
    return True


def semantic_consensus(
    run_structures: list,
    embedder: ClaimEmbedder,
    similarity_threshold: float = 0.84,
    min_agreement: float = 0.6,
    verbose: bool = True,
) -> list:
    records = []
    for run_idx, structure in enumerate(run_structures):
        if verbose:
            run_claim_count = sum(len(structure.get(t, [])) for t in CLAIM_TYPES)
            print(f"[run {run_idx}] 추출된 claim {run_claim_count}개")
            for claim_type in CLAIM_TYPES:
                for item in structure.get(claim_type, []):
                    print(f"    [{claim_type}] {item['text']}")
        for claim_type in CLAIM_TYPES:
            for item in structure.get(claim_type, []):
                rec = dict(item)
                rec["type"] = claim_type
                rec["run_idx"] = run_idx
                records.append(rec)
    if not records:
        if verbose:
            print("[consensus] 모든 run에서 claim이 추출되지 않았습니다.")
        return []

    if verbose:
        print(f"\n[embedding] 전체 claim {len(records)}개 임베딩 계산 중...")
    embeddings = embedder.encode([r["text"] for r in records])

    clusters = []
    for idx, rec in enumerate(records):
        best_cluster, best_sim = None, -1.0
        for cluster in clusters:
            if not claims_compatible(rec, cluster["prototype"]):
                continue
            sim = float(np.dot(embeddings[idx], cluster["centroid"]))
            if sim > best_sim:
                best_cluster, best_sim = cluster, sim
        if best_cluster is not None and best_sim >= similarity_threshold:
            cluster_idx = clusters.index(best_cluster)
            best_cluster["indices"].append(idx)
            vecs = embeddings[best_cluster["indices"]]
            centroid = vecs.mean(axis=0)
            best_cluster["centroid"] = centroid / max(np.linalg.norm(centroid), 1e-12)
            if verbose:
                print(f"  [cluster #{cluster_idx}] run{rec['run_idx']} \"{rec['text']}\" 합류 (유사도 {best_sim:.4f} >= {similarity_threshold})")
        else:
            clusters.append({
                "prototype": rec,
                "indices": [idx],
                "centroid": embeddings[idx].copy(),
            })
            if verbose:
                reason = f"최고 유사도 {best_sim:.4f} < {similarity_threshold}" if best_cluster is not None else "비교 가능한 클러스터 없음"
                print(f"  [cluster #{len(clusters)-1}] run{rec['run_idx']} \"{rec['text']}\" 신규 생성 ({reason})")

    n_runs = len(run_structures)
    output = []
    if verbose:
        print(f"\n[consensus] 클러스터 {len(clusters)}개 생성됨. min_agreement={min_agreement} 기준으로 채택 여부 판단.")
    for cluster_idx, cluster in enumerate(clusters):
        idxs = cluster["indices"]
        members = [records[i] for i in idxs]
        run_ids = sorted(set(m["run_idx"] for m in members))
        votes = len(run_ids)
        agreement = votes / n_runs

        vecs = embeddings[idxs]
        sims = cosine_similarity(vecs)

        if agreement < min_agreement:
            if verbose:
                print(f"  클러스터 #{cluster_idx}: {votes}/{n_runs} run 참여 (agreement={agreement:.2f}) -> 기각 (< {min_agreement})")
            continue

        representative_idx = int(np.argmax(sims.mean(axis=1)))
        representative = dict(members[representative_idx])
        representative.pop("run_idx", None)
        representative.update({
            "claim_id": f"c{len(output)+1}",
            "agreement": round(agreement, 4),
            "votes": votes,
            "total_runs": n_runs,
            "mean_semantic_similarity": round(float(sims.mean()), 4),
            "variants": sorted(set(m["text"] for m in members)),
            "source_spans": sorted(set(str(m.get("source_span", "")) for m in members if m.get("source_span"))),
        })
        if verbose:
            print(
                f"  클러스터 #{cluster_idx}: {votes}/{n_runs} run 참여 (agreement={agreement:.2f}) -> 채택: "
                f"{representative['claim_id']} = \"{representative['text']}\" (variants: {representative['variants']})"
            )
        output.append(representative)

    if verbose:
        print(f"\n[consensus 완료] 최종 claim {len(output)}개\n")
    return output


def extract_consensus_claims(
    extractor,
    embedder: ClaimEmbedder,
    description: str,
    extraction_config: dict,
    return_runs: bool = False,
    verbose: bool = True,
):
    start = time.perf_counter()
    n_runs = extraction_config["n_runs"]
    if verbose:
        print(f"=== {n_runs}개 run을 배치로 한 번에 생성 중 (seed={extraction_config['base_seed']}) ===")
    runs = extractor.extract_runs(
        description=description,
        n_runs=n_runs,
        seed=extraction_config["base_seed"],
        temperature=extraction_config["temperature"],
        top_p=extraction_config["top_p"],
        max_new_tokens=extraction_config["max_new_tokens"],
    )
    if verbose:
        print(f"\n=== {len(runs)}회 추출 완료. 임베딩 기반 의미 합의(consensus) 계산 시작 ===")
    claims = semantic_consensus(
        runs,
        embedder,
        similarity_threshold=extraction_config["similarity_threshold"],
        min_agreement=extraction_config["min_agreement"],
        verbose=verbose,
    )
    elapsed = time.perf_counter() - start
    if return_runs:
        return claims, runs, elapsed
    return claims

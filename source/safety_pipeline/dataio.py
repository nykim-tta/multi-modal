"""공유 입출력 — 데이터셋 로드, A 산출물(claim 집합) 저장/로드.

A ↔ B 접점: A 는 save_claim_sets() 로 claim 집합을 디스크에 남기고,
B 는 load_claim_sets() 로 읽어서 검증합니다. (두 단계를 다른 시점/PC 에서 실행 가능)
"""

import json
from pathlib import Path

from .records import ClaimSet, EvaluationRecord


def load_samples_from_data_dir(data_dir: str) -> list:
    """data 폴더의 *_gold.json(우선) 또는 *_processed.json 을 읽어 samples 리스트를 만든다.

    - gold standard 세트(data2): <folder>_gold.json
    - 일반 세트(data): <folder>_processed.json
    각 샘플 = {sample_id, image_path, description}
    """
    data_path = Path(data_dir)
    json_paths = sorted(data_path.rglob("*_gold.json")) or sorted(data_path.rglob("*_processed.json"))
    samples = []
    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            item = json.load(f)
        index = item.get("index", json_path.stem)
        for img_entry in item.get("image_info", {}).get("image_report", []):
            image_path = json_path.parent / img_entry["image_path"]
            samples.append({
                "sample_id": f"{index}_{img_entry['image_index']}",
                "image_path": str(image_path),
                "description": img_entry["description"],
            })
    return samples


def stage_a_filename(claim_model: str, embedding_model: str) -> str:
    return f"stageA_claims__{claim_model}__{embedding_model}.json"


def save_claim_sets(claim_sets: list, path) -> Path:
    """list[ClaimSet] 를 JSON 으로 저장한다 (A 산출물)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [cs.to_dict() for cs in claim_sets]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_claim_sets(path) -> list:
    """A 가 저장한 claim 집합 JSON 을 list[ClaimSet] 로 복원한다 (B 입력)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ClaimSet.from_dict(d) for d in data]


def load_all_claim_sets(stage_a_dir) -> dict:
    """Stage A 디렉터리의 모든 stageA_claims__*.json 을 읽어 조합별 dict 로 반환한다.

    반환: {(claim_model, embedding_model): list[ClaimSet]}  (stage_b.run_stage_b 입력 형식)
    """
    stage_a_dir = Path(stage_a_dir)
    combos = {}
    for path in sorted(stage_a_dir.glob("stageA_claims__*.json")):
        stem = path.stem[len("stageA_claims__"):]   # "<claim_model>__<embedding_model>"
        if "__" not in stem:
            continue
        claim_model, embedding_model = stem.split("__", 1)
        combos[(claim_model, embedding_model)] = load_claim_sets(path)
    return combos


def save_json(obj, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

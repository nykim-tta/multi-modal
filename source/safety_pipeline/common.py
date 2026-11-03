"""공유 유틸리티 — 시드/메모리, dtype 변환, LLM/VLM 출력 JSON 파싱, 안정적 ID."""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import re
import json
import hashlib

import torch


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_dtype(name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(name, torch.bfloat16)


def _repair_json(s: str) -> str:
    """로컬 LLM이 자주 만드는 JSON 문법 오류를 보정한다."""
    # 1) 후행 쉼표 제거:  "..."],  }  ->  "..."] }
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    # 2) 스마트 따옴표를 표준 따옴표로 치환
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return s


def extract_json(text: str):
    """모델 원문 출력에서 코드펜스/잡담을 제거하고 JSON을 파싱한다.

    1차 파싱이 실패하면 흔한 문법 오류(후행 쉼표, 스마트 따옴표)를 보정 후 재시도.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    starts = [i for i in (cleaned.find("["), cleaned.find("{")) if i >= 0]
    ends = [i for i in (cleaned.rfind("]"), cleaned.rfind("}")) if i >= 0]
    if starts and ends:
        cleaned = cleaned[min(starts):max(ends) + 1]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return json.loads(_repair_json(cleaned))


def stable_id(*parts: str) -> str:
    raw = "||".join(str(x) for x in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

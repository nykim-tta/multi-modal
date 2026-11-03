"""[담당 B] VLM 검증 — 검증 프롬프트 + VLM 어댑터.

다른 VLM 을 비교하려면 verify(image_path, claims) -> list[verdict dict] 를 제공하는
어댑터 클래스를 추가하고 load_vlm_verifier() 의 family 분기에 등록하면 됩니다.
"""

import json
from typing import Any

import torch

from .common import resolve_dtype, clear_memory, extract_json


VERIFY_SYSTEM_PROMPT = r"""
당신은 이미지와 atomic claim 목록을 대조하는 엄격한 시각 검증자입니다.
각 claim을 독립적으로 판단하세요.

판정값:
- supported: 이미지에서 명확히 확인됨
- contradicted: 이미지와 명백히 모순됨
- uncertain: 이미지 해상도, 각도, 가림 또는 시각적 비판별성 때문에 확인할 수 없음

주의:
- 이미지에서 보이지 않는다는 이유만으로 contradicted로 판정하지 마세요.
- Not visible is not the same as false.
- claim 문장에 포함되지 않은 정보를 추론하지 마세요.

반드시 아래 JSON 배열만 출력하세요:
[
  {
    "claim_id": "c1",
    "verdict": "supported" | "contradicted" | "uncertain",
    "support_score": 0.0부터 1.0 사이의 수,
    "evidence": "짧은 시각적 근거"
  }
]
"""


class Qwen25VLVerifier:
    def __init__(self, name: str, config: dict):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.name = name
        self.config = config
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config["model_id"],
            torch_dtype=resolve_dtype(config.get("dtype", "bfloat16")),
            device_map=config.get("device_map", "auto"),
        )
        self.processor = AutoProcessor.from_pretrained(
            config["model_id"],
            min_pixels=config.get("min_pixels"),
            max_pixels=config.get("max_pixels"),
        )

    def verify(self, image_path: str, claims: list, max_new_tokens: int = 1024) -> list:
        from qwen_vl_utils import process_vision_info
        if not claims:
            return []
        compact_claims = [
            {"claim_id": c["claim_id"], "type": c["type"], "text": c["text"]}
            for c in claims
        ]
        messages = [
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "다음 claim을 검증하세요:\n" + json.dumps(compact_claims, ensure_ascii=False)},
            ]},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = generated[0, inputs.input_ids.shape[1]:]
        raw = self.processor.decode(trimmed, skip_special_tokens=True)
        parsed = extract_json(raw)
        return validate_verdicts(parsed, claims)

    def unload(self):
        del self.model
        del self.processor
        clear_memory()


def validate_verdicts(data: Any, claims: list) -> list:
    if not isinstance(data, list):
        raise ValueError("VLM output must be a JSON array.")
    by_id = {str(x.get("claim_id")): x for x in data if isinstance(x, dict)}
    output = []
    for claim in claims:
        item = by_id.get(claim["claim_id"], {})
        verdict = str(item.get("verdict", "uncertain")).lower()
        if verdict not in {"supported", "contradicted", "uncertain"}:
            verdict = "uncertain"
        raw_score = item.get("support_score")
        if raw_score is None:
            raw_score = 1.0 if verdict == "supported" else 0.0 if verdict == "contradicted" else None
        try:
            support_score = None if raw_score is None else min(1.0, max(0.0, float(raw_score)))
        except Exception:
            support_score = None
        output.append({
            "claim_id": claim["claim_id"],
            "verdict": verdict,
            "support_score": support_score,
            "evidence": str(item.get("evidence", "")),
        })
    return output


def load_vlm_verifier(name: str, config: dict):
    family = config.get("family")
    if family == "qwen2_5_vl":
        return Qwen25VLVerifier(name, config)
    raise NotImplementedError(f"Unsupported VLM family: {family}")

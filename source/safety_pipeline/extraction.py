"""[담당 A] Claim 추출 — 추출 프롬프트 + 텍스트 LLM 어댑터.

다른 API/엔진을 비교하려면 extract_runs() 와 동일한 반환 형식(list[dict], 각 dict는
config.CLAIM_TYPES 키를 갖는 구조)을 구현한 어댑터를 추가하면 됩니다.
"""

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import CLAIM_TYPES
from .common import set_seed, clear_memory, resolve_dtype, extract_json


CLAIM_EXTRACTION_SYSTEM_PROMPT = r"""
You extract atomic, explicitly stated claims from a Korean image-description sentence about a civil safety report.

Return only one valid JSON object using exactly this schema:
{
  "object_identity": [{"text": "...", "source_span": "...", "subject": "..."}],
  "attribute": [{"text": "...", "source_span": "...", "subject": "...", "value": "..."}],
  "action": [{"text": "...", "source_span": "...", "subject": "...", "value": "..."}],
  "location": [{"text": "...", "source_span": "...", "subject": "...", "value": "..."}],
  "temporal": [{"text": "...", "source_span": "...", "subject": "...", "value": "..."}],
  "text_content": [{"text": "...", "source_span": "...", "subject": "...", "value": "..."}],
  "context": [{"text": "...", "source_span": "...", "subject": "...", "value": "..."}]
}

Allowed claim types:
- object_identity: existence, kind, or identity of a target. (예: "SUV 한 대가 있다.")
- attribute: a target's color, condition/state, or numeric property. (예: "SUV는 흰색이다.")
- action: a target's action, motion, or interaction. (예: "SUV가 주차되어 있다.")
- location: the place or position where a target exists or an action occurs, including directional placement (좌/우/앞/뒤). (예: "SUV는 도로 왼쪽에 위치한다.")
- temporal: time or temporal information about the scene. (예: "장면은 밤이다.")
- text_content: text contained in the image — signage, license plates, phrases. (예: "표지판에 '주정차 금지'라고 적혀 있다.")
- context: causal, inferential, or speculative information that is hard to verify from the image alone. (예: "SUV가 신호를 무시했다.")

Boundary rules to avoid overlap:
- attribute vs action: a static property or condition (color, material, damaged, tilted, a number) → attribute; a verb-like doing or interaction (parked, driving, connected, leaning against) → action.
- directional words (좌/우/앞/뒤) belong to location, NOT a separate type.
- temporal (밤/낮/야간) is about the scene; set "subject" to "장면".
- context is for statements that go beyond what a single image can confirm (causation, intent, reasoning). Route such statements here instead of discarding them; also set "subject" to "장면" if it is not about one specific object.

Mandatory procedure:
1. Inspect every allowed claim type.
2. Extract all distinct facts explicitly stated in the input.
3. If no fact exists for a type, return an empty list for that type.
4. Never invent a claim merely to fill a type.

Rules:
- Keep all output values in Korean.
- One claim must contain one independently verifiable fact (one target, one fact).
- Do not paraphrase, soften, or normalize the original wording (keep the caption's own words).
- Attach each attribute/action/location to the object it modifies via "subject".
- Exclude smell, sound, and repair requests.

Example input:
밤에 촬영된 사진으로, 도로 왼쪽에 흰색 SUV 한 대가 주차되어 있고, 그 앞 표지판에는 '주정차 금지'라고 적혀 있다.

Example output:
{
  "object_identity": [
    {"text": "SUV가 있다.", "source_span": "SUV", "subject": "SUV"},
    {"text": "표지판이 있다.", "source_span": "표지판", "subject": "표지판"}
  ],
  "attribute": [
    {"text": "SUV는 흰색이다.", "source_span": "흰색", "subject": "SUV", "value": "흰색"}
  ],
  "action": [
    {"text": "SUV가 주차되어 있다.", "source_span": "주차되어", "subject": "SUV", "value": "주차"}
  ],
  "location": [
    {"text": "SUV가 도로 왼쪽에 있다.", "source_span": "도로 왼쪽에", "subject": "SUV", "value": "도로 왼쪽"},
    {"text": "표지판이 SUV 앞에 있다.", "source_span": "그 앞", "subject": "표지판", "value": "SUV 앞"}
  ],
  "temporal": [
    {"text": "장면은 밤이다.", "source_span": "밤에", "subject": "장면", "value": "밤"}
  ],
  "text_content": [
    {"text": "표지판에 '주정차 금지'라고 적혀 있다.", "source_span": "'주정차 금지'", "subject": "표지판", "value": "주정차 금지"}
  ],
  "context": []
}
"""


class TransformersClaimExtractor:
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        trust = config.get("trust_remote_code", False)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_id"], trust_remote_code=trust,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            torch_dtype=resolve_dtype(config.get("dtype", "bfloat16")),
            device_map=config.get("device_map", "auto"),
            trust_remote_code=trust,
        )

    def _build_prompt(self, description):
        """system 역할을 지원하는 모델은 그대로, 아닌 모델(gemma 등)은 system을 user에 합쳐 프롬프트 생성."""
        messages = [
            {"role": "system", "content": CLAIM_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # gemma 등 system role 미지원 → 시스템 프롬프트를 user 메시지 앞에 병합
            merged = [{"role": "user",
                       "content": CLAIM_EXTRACTION_SYSTEM_PROMPT.strip() + "\n\n" + description}]
            return self.tokenizer.apply_chat_template(
                merged, tokenize=False, add_generation_prompt=True
            )

    def _generate_batch(self, description, num_return_sequences, seed, temperature, top_p, max_new_tokens):
        """동일 프롬프트로 num_return_sequences개의 샘플을 한 번의 generate 호출로 생성한다."""
        set_seed(seed)
        prompt = self._build_prompt(description)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            kwargs.update({"temperature": temperature, "top_p": top_p})
        with torch.no_grad():
            output = self.model.generate(**inputs, **kwargs)
        input_len = inputs["input_ids"].shape[1]
        return [
            self.tokenizer.decode(output[i, input_len:], skip_special_tokens=True)
            for i in range(output.shape[0])
        ]

    def extract_runs(
        self,
        description: str,
        n_runs: int,
        seed: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        max_rounds: int = 3,
    ) -> list:
        """n_runs개의 독립 샘플을 배치로 생성하고 파싱에 성공한 claim 구조 리스트를 반환한다."""
        successes = []
        round_idx = 0
        while len(successes) < n_runs and round_idx < max_rounds:
            need = n_runs - len(successes)
            raws = self._generate_batch(
                description, need, seed + round_idx * 1000, temperature, top_p, max_new_tokens
            )
            for raw in raws:
                try:
                    successes.append(validate_claim_structure(extract_json(raw)))
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"    [경고] JSON 파싱 실패 (round {round_idx + 1}): {e}")
            round_idx += 1
        if not successes:
            raise RuntimeError(f"[{self.name}] claim 추출 JSON 파싱에 모두 실패했습니다.")
        if len(successes) < n_runs:
            print(f"    [알림] 요청한 {n_runs}회 중 {len(successes)}회만 파싱 성공 → {len(successes)}개로 합의를 계산합니다.")
        return successes

    def extract_once(
        self,
        description: str,
        seed: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        max_retries: int = 3,
    ) -> dict:
        """단일 샘플 추출 (배치가 아닌 개별 호출이 필요할 때 사용)."""
        last_err = None
        raw = ""
        for attempt in range(max_retries):
            raw = self._generate_batch(
                description, 1, seed + attempt * 1000, temperature, top_p, max_new_tokens
            )[0]
            try:
                return validate_claim_structure(extract_json(raw))
            except (json.JSONDecodeError, ValueError) as e:
                last_err = e
                print(f"    [경고] JSON 파싱 실패 (시도 {attempt + 1}/{max_retries}): {e}")
        raise RuntimeError(
            f"[{self.name}] claim 추출 JSON 파싱 {max_retries}회 실패: {last_err}\n"
            f"--- 마지막 모델 원문 출력 ---\n{raw}\n---------------------------"
        )

    def unload(self):
        del self.model
        del self.tokenizer
        clear_memory()


def validate_claim_structure(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Claim extraction output must be a JSON object.")
    out = {}
    for claim_type in CLAIM_TYPES:
        items = data.get(claim_type, [])
        if not isinstance(items, list):
            items = []
        clean_items = []
        for item in items:
            if isinstance(item, str):
                item = {"text": item}
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            clean = dict(item)
            clean["text"] = text
            clean["type"] = claim_type
            clean_items.append(clean)
        out[claim_type] = clean_items
    return out

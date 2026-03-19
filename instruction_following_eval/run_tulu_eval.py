#!/usr/bin/env python3
"""
将 tulu-3-sft-personas-instruction-following.parquet 转为 IFEval 格式并运行评估.

Parquet 的 constraints 与 IFEval instruction_id 的映射关系,
kwargs 从 prompt 文本中用正则提取.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

# Parquet constraint -> IFEval instruction_id
CONSTRAINT_TO_INSTRUCTION = {
    "keywords:exclude words": "keywords:forbidden_words",
    "format:title": "detectable_format:title",
    "keywords:letter frequency": "keywords:letter_frequency",
    "format:number of bullet lists": "detectable_format:number_bullet_lists",
    "length constraints:number of paragraphs": "length_constraints:number_paragraphs",
    "case:in english and lowercase": "change_case:english_lowercase",
    "in english and capital": "change_case:english_capital",
    "length constraints:number of sentences": "length_constraints:number_sentences",
    "keywords:frequency": "keywords:frequency",
    "content:include a postscript": "detectable_content:postscript",
    "content:number of placeholders": "detectable_content:number_placeholders",
    "format:choose one from options": "detectable_format:constrained_response",
    "format:number of highlighted sections": "detectable_format:number_highlighted_sections",
    "format:number of sections": "detectable_format:multiple_sections",
    "format:use json format": "detectable_format:json_format",
    "case: frequency of capital words": "change_case:capital_word_frequency",
    "include keywords": "keywords:existence",
    "length constraints:first word of the nth paragraph": "length_constraints:nth_paragraph_first_word",
    "length constraints:number of words": "length_constraints:number_words",
    "punctuation:use no comma": "punctuation:no_comma",
    "repeat the prompt": "combination:repeat_prompt",
    "response language": "language:response_language",
    "specific ending": "startend:end_checker",
    "use quotation": "startend:quotation",
    " give two responses": "combination:two_responses",
}

# IFEval 支持的 instruction_id 集合
VALID_INSTRUCTIONS = {
    "punctuation:no_comma",
    "detectable_format:number_highlighted_sections",
    "length_constraints:number_words",
    "change_case:english_lowercase",
    "change_case:english_capital",
    "change_case:capital_word_frequency",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "detectable_format:number_bullet_lists",
    "detectable_format:multiple_sections",
    "detectable_format:json_format",
    "detectable_format:title",
    "detectable_format:constrained_response",
    "keywords:existence",
    "keywords:forbidden_words",
    "keywords:frequency",
    "keywords:letter_frequency",
    "language:response_language",
    "length_constraints:number_paragraphs",
    "length_constraints:number_sentences",
    "length_constraints:nth_paragraph_first_word",
    "combination:repeat_prompt",
    "combination:two_responses",
    "startend:end_checker",
    "startend:quotation",
}


def extract_kwargs(instruction_id: str, prompt: str) -> dict | None:
    """从 prompt 文本中提取 kwargs. 无法提取时返回 None (跳过该指令)."""
    prompt_lower = prompt.lower()

    if instruction_id == "detectable_content:number_placeholders":
        m = re.search(r"(\d+)\s*placeholder", prompt_lower)
        if m:
            return {"num_placeholders": int(m.group(1))}
        m = re.search(r"at least\s+(\d+)\s*placeholder", prompt_lower)
        if m:
            return {"num_placeholders": int(m.group(1))}

    if instruction_id == "detectable_format:number_bullet_lists":
        m = re.search(r"(\d+)\s*bullet", prompt_lower)
        if m:
            return {"num_bullets": int(m.group(1))}
        m = re.search(r"(\d+)\s*bullet\s*list", prompt_lower)
        if m:
            return {"num_bullets": int(m.group(1))}

    if instruction_id == "detectable_format:number_highlighted_sections":
        m = re.search(r"(\d+)\s*highlight", prompt_lower)
        if m:
            return {"num_highlights": int(m.group(1))}
        m = re.search(r"at least\s+(\d+)\s*section", prompt_lower)
        if m:
            return {"num_highlights": int(m.group(1))}

    if instruction_id == "detectable_format:multiple_sections":
        m = re.search(r"(\d+)\s*section", prompt_lower)
        if m:
            return {"section_spliter": "SECTION", "num_sections": int(m.group(1))}
        m = re.search(r"(\d+)\s*paragraph", prompt_lower)
        if m:
            return {"section_spliter": "PARAGRAPH", "num_sections": int(m.group(1))}

    if instruction_id == "length_constraints:number_paragraphs":
        m = re.search(r"(\d+)\s*paragraph", prompt_lower)
        if m:
            return {"num_paragraphs": int(m.group(1))}

    if instruction_id == "length_constraints:number_sentences":
        m = re.search(r"(\d+)\s*(?:or\s+)?(?:less|fewer)\s+words", prompt_lower)
        if m:
            return {"relation": "less than", "num_sentences": int(m.group(1))}
        m = re.search(r"(?:less|fewer)\s+than\s+(\d+)\s*sentence", prompt_lower)
        if m:
            return {"relation": "less than", "num_sentences": int(m.group(1))}
        m = re.search(r"at least\s+(\d+)\s*sentence", prompt_lower)
        if m:
            return {"relation": "at least", "num_sentences": int(m.group(1))}
        m = re.search(r"(\d+)\s*sentence", prompt_lower)
        if m:
            return {"relation": "at least", "num_sentences": int(m.group(1))}

    if instruction_id == "length_constraints:number_words":
        m = re.search(r"(\d+)\s*(?:or\s+)?(?:less|fewer)\s+words", prompt_lower)
        if m:
            return {"relation": "less than", "num_words": int(m.group(1))}
        m = re.search(r"(?:less|fewer)\s+than\s+(\d+)\s*words", prompt_lower)
        if m:
            return {"relation": "less than", "num_words": int(m.group(1))}
        m = re.search(r"at least\s+(\d+)\s*words", prompt_lower)
        if m:
            return {"relation": "at least", "num_words": int(m.group(1))}
        m = re.search(r"(\d+)\+?\s*words", prompt_lower)
        if m:
            return {"relation": "at least", "num_words": int(m.group(1))}
        m = re.search(r"(\d+)\s*words", prompt_lower)
        if m:
            return {"relation": "at least", "num_words": int(m.group(1))}

    if instruction_id == "length_constraints:nth_paragraph_first_word":
        m = re.search(r"(\d+)\s*paragraph", prompt_lower)
        num_paragraphs = int(m.group(1)) if m else 4
        m = re.search(r"start\s+(?:the\s+)?(?:first|1st|second|2nd|third|3rd|fourth|4th)\s+paragraph\s+with\s+(?:the\s+word\s+)?[\"']([^\"']+)[\"']", prompt_lower)
        if m:
            return {"first_word": m.group(1).strip(), "num_paragraphs": num_paragraphs, "nth_paragraph": 1}
        m = re.search(r"paragraph\s+(\d+)\s+must\s+start\s+with\s+(?:the\s+word\s+)?[\"']([^\"']+)[\"']", prompt_lower)
        if m:
            return {"first_word": m.group(2).strip(), "num_paragraphs": num_paragraphs, "nth_paragraph": int(m.group(1))}
        m = re.search(r"start\s+.*?with\s+(?:the\s+word\s+)?[\"']([^\"']+)[\"']", prompt_lower)
        if m:
            return {"first_word": m.group(1).strip(), "num_paragraphs": num_paragraphs, "nth_paragraph": 1}

    if instruction_id == "keywords:forbidden_words":
        m = re.search(r"exclude\s+(?:the\s+words?\s+)?[\"']([^\"']+)[\"']", prompt_lower)
        if m:
            words = [w.strip() for w in m.group(1).split(",")]
            return {"forbidden_words": words}
        m = re.search(r"exclude\s+([\w\s,]+?)(?:\.|$)", prompt_lower)
        if m:
            words = [w.strip() for w in re.split(r"[\s,]+", m.group(1)) if w.strip() and len(w.strip()) > 2]
            if words:
                return {"forbidden_words": words[:5]}

    if instruction_id == "keywords:existence":
        m = re.search(r"include\s+(?:the\s+)?keywords?\s*:?\s*[\"']([^\"']+)[\"']", prompt_lower)
        if m:
            words = [w.strip() for w in m.group(1).split(",")]
            return {"keywords": words}
        m = re.search(r"include\s+([\w\s,]+?)(?:\.|$)", prompt_lower)
        if m:
            words = [w.strip() for w in re.split(r"[\s,]+", m.group(1)) if w.strip()]
            if words:
                return {"keywords": words[:5]}

    if instruction_id == "keywords:frequency":
        m = re.search(r"[\"'](\w+)[\"']\s+(?:should\s+)?(?:appear|occur)\s+(?:at least\s+)?(\d+)", prompt_lower)
        if m:
            return {"relation": "at least", "keyword": m.group(1), "frequency": int(m.group(2))}
        m = re.search(r"(\w+)\s+(?:at least\s+)?(\d+)", prompt_lower)
        if m and m.group(1) not in ("paragraph", "sentence", "word", "bullet", "section"):
            return {"relation": "at least", "keyword": m.group(1), "frequency": int(m.group(2))}

    if instruction_id == "keywords:letter_frequency":
        m = re.search(r"[\"']([#!]|.)[\"']\s+(?:at least\s+)?(\d+)", prompt_lower)
        if m:
            return {"let_relation": "at least", "letter": m.group(1), "let_frequency": int(m.group(2))}
        m = re.search(r"(\d+)\s*(?:or\s+)?(?:more\s+)?(?:exclamation|hashtag|#)", prompt_lower)
        if m:
            return {"let_relation": "at least", "letter": "#", "let_frequency": int(m.group(1))}
        m = re.search(r"(\d+)\s*[!]", prompt_lower)
        if m:
            return {"let_relation": "at least", "letter": "!", "let_frequency": int(m.group(1))}

    if instruction_id == "change_case:capital_word_frequency":
        m = re.search(r"capital\s+(?:words?\s+)?(?:less than|fewer than)\s+(\d+)", prompt_lower)
        if m:
            return {"capital_relation": "less than", "capital_frequency": int(m.group(1))}
        m = re.search(r"capital\s+(?:words?\s+)?(?:at least\s+)?(\d+)", prompt_lower)
        if m:
            return {"capital_relation": "at least", "capital_frequency": int(m.group(1))}

    if instruction_id == "detectable_content:postscript":
        if "p.s." in prompt_lower or "p.s" in prompt_lower:
            return {"postscript_marker": "P.S."}
        return {"postscript_marker": "P.S."}

    if instruction_id == "language:response_language":
        # 仅支持 IFEval instructions_util.LANGUAGE_CODES 中的语言
        lang_map = {
            "hindi": "hi", "spanish": "es", "french": "fr", "german": "de",
            "japanese": "ja", "korean": "ko", "arabic": "ar", "portuguese": "pt",
            "russian": "ru", "italian": "it", "bengali": "bn", "thai": "th",
            "urdu": "ur", "tamil": "ta", "telugu": "te", "bulgarian": "bg",
            "persian": "fa", "vietnamese": "vi", "nepali": "ne", "swahili": "sw",
            "kannada": "kn", "marathi": "mr", "gujarati": "gu", "punjabi": "pa",
            "finnish": "fi",
        }
        for name, code in lang_map.items():
            if name in prompt_lower:
                return {"language": code}
        return None  # 无法识别或不在支持列表则跳过

    if instruction_id == "combination:repeat_prompt":
        user_part = prompt.split(".")[0] if "." in prompt else prompt
        repeat_part = user_part[:500]
        return {"prompt_to_repeat": repeat_part}

    if instruction_id == "startend:end_checker":
        m = re.search(r"end\s+(?:with|your response with)\s+[\"']([^\"']+)[\"']", prompt_lower)
        if m:
            return {"end_phrase": m.group(1).strip()}

    return {}


def needs_kwargs(instruction_id: str) -> bool:
    """该指令是否需要非空 kwargs."""
    no_kwargs = {
        "punctuation:no_comma",
        "change_case:english_lowercase",
        "change_case:english_capital",
        "detectable_format:title",
        "detectable_format:json_format",
        "detectable_format:constrained_response",
        "combination:two_responses",
        "startend:quotation",
    }
    return instruction_id not in no_kwargs


def convert_row(row, key: int) -> tuple[dict, dict] | None:
    """将 parquet 一行转为 input_data 和 input_response_data 条目. 无法转换时返回 None."""
    prompt = row["prompt"]
    constraints = row["constraints"]
    if hasattr(constraints, "tolist"):
        constraints = constraints.tolist()

    messages = row["messages"]
    response = ""
    for m in messages:
        if m.get("role") == "assistant":
            response = m.get("content", "")
            break

    if not response.strip():
        return None

    instruction_id_list = []
    kwargs_list = []

    for c in constraints:
        c_str = c.strip() if isinstance(c, str) else str(c).strip()
        inst_id = CONSTRAINT_TO_INSTRUCTION.get(c_str)
        if inst_id is None or inst_id not in VALID_INSTRUCTIONS:
            continue

        kw = extract_kwargs(inst_id, prompt)
        if needs_kwargs(inst_id) and kw is None:
            continue
        if kw is None:
            kw = {}

        instruction_id_list.append(inst_id)
        kwargs_list.append(kw)

    if not instruction_id_list:
        return None

    input_entry = {
        "key": key,
        "prompt": prompt,
        "instruction_id_list": instruction_id_list,
        "kwargs": kwargs_list,
    }
    response_entry = {"prompt": prompt, "response": response}
    return (input_entry, response_entry)


def main():
    data_dir = Path(__file__).parent / "data"
    parquet_path = data_dir / "tulu-3-sft-personas-instruction-following.parquet"
    if not parquet_path.exists():
        print(f"File not found: {parquet_path}")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    input_entries = []
    response_entries = []

    for i, row in df.iterrows():
        result = convert_row(row, key=i)
        if result:
            input_entries.append(result[0])
            response_entries.append(result[1])

    input_path = data_dir / "tulu_input_data.jsonl"
    response_path = data_dir / "tulu_input_response_data.jsonl"

    with open(input_path, "w") as f:
        for e in input_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    with open(response_path, "w") as f:
        for e in response_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Converted {len(input_entries)} samples (from {len(df)} total)")
    print(f"  input_data: {input_path}")
    print(f"  input_response_data: {response_path}")
    return input_path, response_path


if __name__ == "__main__":
    main()

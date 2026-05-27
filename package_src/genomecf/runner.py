from __future__ import annotations


def normalize_mode(mode: str) -> str:
    value = mode.strip().lower()
    mapping = {
        "diagnostic": "frozen",
        "classical": "frozen",
        "supervised": "full",
        "full": "full",
        "frozen": "frozen",
        "lora": "lora",
    }
    if value not in mapping:
        raise ValueError(f"Unsupported mode: {mode}")
    return mapping[value]

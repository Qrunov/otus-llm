#!/usr/bin/env python3
"""Load first N rows from hugsid/legal-contracts (train) and add empty entity arrays."""

import json
from pathlib import Path

from datasets import load_dataset

ENTITY_KEYS = (
    "PERSON",
    "ORG",
    "MONEY",
    "DATE",
    "CONTRACT_TYPE",
    "OBLIGATION",
    "JURISDICTION",
)

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "legal_contracts_train_1k_entities.json"


def main() -> None:
    n = 1000
    ds = load_dataset("hugsid/legal-contracts", split=f"train[:{n}]")
    rows = []
    for row in ds:
        item = dict(row)
        for k in ENTITY_KEYS:
            item[k] = []
        rows.append(item)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(rows)} records to {OUT_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify that sources.bib parses correctly and contains required entries."""

import re
import sys

REQUIRED_KEYS = [
    "lample2020deep",
    "udrescu2020ai",
    "arc2025architects",
    "raissi2019physics",
    "nie2025llada",
    "cranmer2023interpretable",
    "lewkowycz2022solving",
    "azerbayev2024llemma",
    "vaswani2017attention",
    "su2024roformer",
]

REQUIRED_CATEGORIES = {
    "symbolic_math": ["lample2020deep", "kamienny2022end"],
    "ai_feynman": ["udrescu2020ai", "udrescu2020ai2"],
    "arc_2025": ["arc2025architects"],
    "pinns": ["raissi2019physics"],
    "masked_diffusion": ["nie2025llada", "sahoo2024simple"],
    "symbolic_regression": ["cranmer2023interpretable", "koza1994genetic"],
    "llm_math": ["lewkowycz2022solving", "azerbayev2024llemma"],
    "transformers": ["vaswani2017attention", "su2024roformer"],
}


def parse_bib(filepath):
    """Parse BibTeX file and return list of entry keys."""
    with open(filepath, 'r') as f:
        content = f.read()

    entries = re.findall(r'@\w+\{(\w+),', content)
    return entries


def main():
    bib_path = "sources.bib"
    entries = parse_bib(bib_path)

    print(f"Found {len(entries)} BibTeX entries in {bib_path}")
    for e in entries:
        print(f"  - {e}")

    # Check required entries
    missing = [k for k in REQUIRED_KEYS if k not in entries]
    if missing:
        print(f"\nWARNING: Missing required entries: {missing}")
    else:
        print(f"\nAll {len(REQUIRED_KEYS)} required entries present.")

    # Check categories
    print("\nCategory coverage:")
    all_covered = True
    for cat, keys in REQUIRED_CATEGORIES.items():
        found = [k for k in keys if k in entries]
        status = "OK" if found else "MISSING"
        if not found:
            all_covered = False
        print(f"  {cat}: {status} ({len(found)}/{len(keys)} entries)")

    # Check minimum count
    if len(entries) >= 15:
        print(f"\nMinimum entry count: PASS ({len(entries)} >= 15)")
    else:
        print(f"\nMinimum entry count: FAIL ({len(entries)} < 15)")

    print("\nVerification complete.")
    return 0 if (not missing and len(entries) >= 15 and all_covered) else 1


if __name__ == '__main__':
    sys.exit(main())

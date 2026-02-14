#!/usr/bin/env python3
"""Validate that all BibTeX entries in sources.bib are well-formed.

Checks that every entry has the required fields for its entry type:
  - All types: author, title, year
  - @article: journal
  - @inproceedings: booktitle
  - @book: publisher

Reports any missing required fields and exits with code 1 if errors are found.
"""

import re
import sys
import os

# Required fields per entry type.  Every type requires author, title, year.
# Additional type-specific required fields are listed below.
COMMON_REQUIRED = {"author", "title", "year"}
TYPE_REQUIRED = {
    "article": {"journal"},
    "inproceedings": {"booktitle"},
    "book": {"publisher"},
    "misc": set(),          # misc entries have no extra required fields
    "phdthesis": {"school"},
    "mastersthesis": {"school"},
    "techreport": {"institution"},
}


def parse_bib_entries(text):
    """Yield (entry_type, cite_key, {field: value}) for each BibTeX entry."""
    # Match @type{key, ... }  (handles nested braces in field values)
    entry_pattern = re.compile(
        r"@(\w+)\s*\{([^,]+),", re.IGNORECASE
    )
    pos = 0
    while pos < len(text):
        m = entry_pattern.search(text, pos)
        if m is None:
            break
        entry_type = m.group(1).lower()
        cite_key = m.group(2).strip()

        # Find the matching closing brace for this entry
        brace_depth = 1
        start = m.end()
        i = start
        while i < len(text) and brace_depth > 0:
            if text[i] == "{":
                brace_depth += 1
            elif text[i] == "}":
                brace_depth -= 1
            i += 1
        entry_body = text[start:i - 1]

        # Parse fields: field_name = {value} or field_name = value
        fields = {}
        field_pattern = re.compile(
            r"(\w+)\s*=\s*(?:\{([^}]*(?:\{[^}]*\}[^}]*)*)\}|\"([^\"]*)\"|(\d+))",
            re.DOTALL,
        )
        for fm in field_pattern.finditer(entry_body):
            fname = fm.group(1).lower()
            fvalue = fm.group(2) or fm.group(3) or fm.group(4) or ""
            fields[fname] = fvalue.strip()

        yield entry_type, cite_key, fields
        pos = i


def validate_entry(entry_type, cite_key, fields):
    """Return list of error strings for this entry (empty if valid)."""
    errors = []
    required = set(COMMON_REQUIRED)
    extra = TYPE_REQUIRED.get(entry_type, set())
    required |= extra

    for field in sorted(required):
        if field not in fields or not fields[field]:
            errors.append(
                f"  [{cite_key}] (@{entry_type}): missing required field '{field}'"
            )
    return errors


def main():
    bib_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "sources.bib",
    )
    if not os.path.isfile(bib_path):
        print(f"ERROR: {bib_path} not found")
        sys.exit(1)

    with open(bib_path, "r", encoding="utf-8") as f:
        text = f.read()

    entries = list(parse_bib_entries(text))
    print(f"Found {len(entries)} BibTeX entries in sources.bib\n")

    all_errors = []
    for entry_type, cite_key, fields in entries:
        errs = validate_entry(entry_type, cite_key, fields)
        all_errors.extend(errs)
        status = "OK" if not errs else "ERRORS"
        field_list = ", ".join(sorted(fields.keys()))
        print(f"  @{entry_type}{{{cite_key}}} -- {status}")
        print(f"    fields: {field_list}")
        if errs:
            for e in errs:
                print(f"    {e}")
        print()

    if all_errors:
        print(f"VALIDATION FAILED: {len(all_errors)} error(s) found")
        for e in all_errors:
            print(e)
        sys.exit(1)
    else:
        print("VALIDATION PASSED: All entries have required fields.")
        sys.exit(0)


if __name__ == "__main__":
    main()

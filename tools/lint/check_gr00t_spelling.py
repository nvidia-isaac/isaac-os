#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-commit hook to enforce 'gr00t' spelling (with zeros, not 'groot').

Scans staged files for occurrences of 'groot' that should be 'gr00t'.
Excludes known exceptions such as external URLs, checkpoint artifact names,
and the tsconfigRootDir JavaScript pattern.
"""

from pathlib import Path
import re
import sys

# Patterns that are allowed to contain 'groot' (case-insensitive).
# Each is matched against the full line.
ALLOW_PATTERNS = [
    # JavaScript tsconfigRootDir (false positive from 'RootDir').
    re.compile(r'tsconfigRootDir'),
]

# Regex to find 'groot' with regular o's (case-insensitive), but not 'gr00t'.
GROOT_PATTERN = re.compile(r'(?<![0])groot', re.IGNORECASE)


def is_allowed(line: str) -> bool:
    """Return True if the line contains a known exception."""
    return any(p.search(line) for p in ALLOW_PATTERNS)


def check_file(filepath: str) -> list[tuple[int, str]]:
    """Check a single file for 'groot' misspellings.

    Returns a list of (line_number, line_content) tuples for violations.
    """
    path = Path(filepath)

    violations = []
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except (OSError, UnicodeDecodeError):
        return []

    for line_num, line in enumerate(text.splitlines(), start=1):
        if GROOT_PATTERN.search(line) and not is_allowed(line):
            violations.append((line_num, line.rstrip()))

    return violations


def main() -> int:
    if len(sys.argv) < 2:
        return 0

    found_violations = False

    for filepath in sys.argv[1:]:
        violations = check_file(filepath)
        if violations:
            found_violations = True
            print(f'\n{filepath}:')
            for line_num, line in violations:
                print(f'  line {line_num}: {line}')

    if found_violations:
        print(
            '\nError: Found "groot" (with letter o) which should be "gr00t" (with zeros).'
            '\nIf this is a legitimate exception (e.g. external artifact name), '
            'add it to ALLOW_PATTERNS in tools/pre_commit/check_gr00t_spelling.py.'
        )
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())

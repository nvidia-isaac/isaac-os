#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from datetime import datetime
from fnmatch import fnmatch
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_YEAR = datetime.now().year
REPO_ROOT = Path(__file__).resolve().parents[2]

# File type configuration: maps extensions to comment delimiters.
# Special files can be checked by name pattern.
FILE_TYPE_CONFIG = {
    # Single-line comment types.
    '.py': '#',
    '.cmake': '#',
    '.ctest': '#',
    '.yaml': '#',
    '.yml': '#',
    '.toml': '#',
    '.dockerfile': '#',
    '.sh': '#',
    '.usda': '#',
    '.fbs': '//',
    '.js': '//',
    '.ts': '//',
    # Block comment types.
    '.h': '/*',
    '.hpp': '/*',
    '.cpp': '/*',
    '.cc': '/*',
    '.cu': '/*',
    '.cuh': '/*',
    '.css': '/*',
}

# Shebangs, Docker parser directives and USD layer headers must stay on line 1.
FIRST_LINE_DIRECTIVE_PREFIXES = ('#!', '# syntax=', '# escape=', '#usda ')

# Block comment start delimiter -> end delimiter.
BLOCK_COMMENT_END = {
    '/*': '*/',
}

# Special file name patterns (name, comment_delimiter).
SPECIAL_FILE_PATTERNS = [
    (lambda name: name == 'CMakeLists.txt', '#'),
    (lambda name: name == 'Dockerfile' or name.startswith('Dockerfile.'), '#'),
    (lambda name: name == 'Doxyfile', '#'),
]

# Files that intentionally carry no license header. Any other file type fails the check, so
# new file types must be added to FILE_TYPE_CONFIG or explicitly exempted here.
# Extensionless dotfiles (e.g. .gitignore, .clang-format) are also exempt.
EXEMPT_SUFFIXES = {
    # Documentation, data and legal texts (LICENSE.txt, NOTICE.txt, requirements*.txt).
    '.md',
    '.mdx',
    '.rst',
    '.json',
    '.txt',
    # Patches apply to third-party sources.
    '.patch',
    # Binary and image assets.
    '.gif',
    '.jpg',
    '.jpeg',
    '.png',
    '.svg',
    '.onnx',
}
EXEMPT_NAMES = {'CODE_OF_CONDUCT'}
# Generated or bundled files (repository-relative glob patterns).
EXEMPT_PATH_PATTERNS = [
    'docs/fern/dist/*',
    'tools/foxglove/isaac-to-foxglove-converter/src/generated/*',
    'uv.lock',
]

license_text = 'SPDX-License-Identifier: Apache-2.0\n'

# The Apache-2.0 notice paragraph that older headers carried after the SPDX lines.
APACHE_NOTICE_WORDS = (
    'Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file '
    'except in compliance with the License. You may obtain a copy of the License at '
    'http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to '
    'in writing, software distributed under the License is distributed on an "AS IS" BASIS, '
    'WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License '
    'for the specific language governing permissions and limitations under the License.'
).split()
_COMMENT_MARKER_RE = re.compile(r'^\s*(?:#|//|/\*!?|\*/?)?')


def make_nvidia_copyright_text(years: str) -> str:
    return (
        f'SPDX-FileCopyrightText: Copyright (c) {years} NVIDIA CORPORATION & AFFILIATES. '
        'All rights reserved.'
    )


default_copyright_text = make_nvidia_copyright_text(str(DEFAULT_YEAR))
default_combined_license = '\n'.join([default_copyright_text, license_text])

copyright_regex_pattern = (
    r'SPDX-FileCopyrightText: Copyright \(c\) \d{4}(?:-\d{4})?(?:,)? '
    r'NVIDIA CORPORATION & AFFILIATES\. All rights reserved\.\r?\n'
    r'(?:SPDX-FileCopyrightText: Copyright \(c\) \d{4}.*\n?\r?)*'
)
license_regex_pattern = copyright_regex_pattern + re.escape(license_text)
license_regex = re.compile(license_regex_pattern)

# Basic regex sanity checks.
assert re.compile(copyright_regex_pattern).match(default_copyright_text + '\n'), (
    'Default copyright text not valid'
)
assert license_regex.match(default_combined_license), 'Default license text or regex is not valid'

_NVIDIA_SPDX_RE = re.compile(
    r'^SPDX-FileCopyrightText: Copyright \(c\) (?P<years>\d{4}(?:-\d{4})?)(?:,)? '
    r'NVIDIA CORPORATION & AFFILIATES\.(?: All rights reserved\.)?\s*$'
)


def _compute_years(existing_years: str, current_year: int) -> str:
    start_year = int(existing_years.split('-', 1)[0])
    if current_year <= start_year:
        return str(start_year)
    return f'{start_year}-{current_year}'


def _find_separate_rights_line(lines: list[str], spdx_index: int, prefix: str) -> int | None:
    accepted_rights_lines = {
        prefix.strip() + ' All rights reserved.',
        (prefix + 'All rights reserved.').strip(),
    }
    for j in range(spdx_index + 1, min(len(lines), spdx_index + 4)):
        candidate = lines[j]
        if 'SPDX-License-Identifier' in candidate:
            break
        if 'SPDX-' not in candidate and candidate.strip() in accepted_rights_lines:
            return j
    return None


def _inline_rights_line(lines: list[str], spdx_index: int, rights_index: int, prefix: str) -> None:
    for j in range(rights_index - 1, spdx_index, -1):
        if lines[j].strip() == prefix.strip():
            del lines[j]
            rights_index -= 1
    del lines[rights_index]


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['git', *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _repo_relpath(filepath: Path) -> str | None:
    try:
        return filepath.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None


def _resolve_in_repo(filepath: Path) -> Path:
    """Resolve a path and reject it if it points outside the repository."""
    repo_root = os.path.realpath(REPO_ROOT)
    resolved = os.path.realpath(filepath)
    if os.path.commonpath([repo_root, resolved]) != repo_root:
        raise ValueError(f'File {filepath} is outside the repository')
    return Path(resolved)


def _write_lines(filepath: Path, lines: list[str]) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def _git_is_staged(relpath: str) -> bool:
    proc = _run_git(['diff', '--cached', '--name-only', '--', relpath])
    return proc.returncode == 0 and proc.stdout.strip() != ''


def _git_last_changed_year(relpath: str) -> int | None:
    proc = _run_git(['log', '-1', '--format=%cd', '--date=format:%Y', '--', relpath])
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    if not out.isdigit():
        return None
    return int(out)


def _effective_current_year(filepath: Path) -> int:
    """Return the year the file should use as the "current year".

    - For staged files (modified or added), use the present year (the year of the
      commit we're about to create).
    - Otherwise, use the year of the last commit that modified the file.
    - If the file is not known to git yet, fall back to the present year.
    """
    now_year = datetime.now().year
    relpath = _repo_relpath(filepath)
    if relpath is None:
        return now_year

    if _git_is_staged(relpath):
        return now_year

    last_changed_year = _git_last_changed_year(relpath)
    return last_changed_year if last_changed_year is not None else now_year


def _normalize_and_update_nvidia_copyrights(
    lines: list[str],
    current_year: int,
    dry_run: bool,
) -> bool:
    """Ensure NVIDIA copyright lines use ranges and inline "All rights reserved.".

    Rules:
    - New files use a single year.
    - Edited files use a range, where the first year stays as-is and the second year
      is the file's current_year.
    - "All rights reserved." must be on the NVIDIA copyright line.
    """
    changed = False
    i = 0
    while i < len(lines):
        changed |= _normalize_nvidia_copyright_line(lines, i, current_year, dry_run)
        i += 1

    return changed


def _normalize_nvidia_copyright_line(
    lines: list[str], i: int, current_year: int, dry_run: bool
) -> bool:
    line = lines[i]
    if 'SPDX-FileCopyrightText:' not in line:
        return False

    idx = line.find('SPDX-FileCopyrightText:')
    prefix = line[:idx]
    match = _NVIDIA_SPDX_RE.match(line[idx:])
    if match is None:
        return False

    new_years = _compute_years(match.group('years'), current_year)
    desired_spdx_line = (
        prefix
        + f'SPDX-FileCopyrightText: Copyright (c) {new_years} NVIDIA CORPORATION & AFFILIATES. '
        + 'All rights reserved.'
    )
    separate_rights_index = _find_separate_rights_line(lines, i, prefix)
    if line == desired_spdx_line and separate_rights_index is None:
        return False
    if not dry_run:
        lines[i] = desired_spdx_line
        if separate_rights_index is not None:
            _inline_rights_line(lines, i, separate_rights_index, prefix)
    return True


def _extract_license_block(
    lines: list[str],
    start_line: int,
    comment_start: str,
    is_multiline_comment: bool,
) -> tuple[list[str], int, int]:
    """Return (license_block_lines, start_index, end_index_exclusive).

    If no comment/license block exists at `start_line`, returns ([], start_line, start_line).
    """
    if is_multiline_comment:
        # Handle block comments such as /* */ (C++, CSS).
        comment_end = BLOCK_COMMENT_END[comment_start]
        if start_line >= len(lines) or not lines[start_line].strip().startswith(comment_start):
            return ([], start_line, start_line)

        license_block: list[str] = []
        for i, line in enumerate(lines[start_line:], start_line):
            license_block.append(line)
            if line.strip().endswith(comment_end):
                return (license_block, start_line, i + 1)
        return (license_block, start_line, len(lines))

    # Handle single-line comments (Python, CMake, etc)
    if start_line >= len(lines) or not lines[start_line].startswith(comment_start):
        return ([], start_line, start_line)

    end = start_line
    for line in lines[start_line:]:
        if not line.startswith(comment_start):
            break
        end += 1
    return (lines[start_line:end], start_line, end)


def _license_block_to_text(
    license_block: list[str],
    comment_start: str,
    is_multiline_comment: bool,
) -> str:
    """Convert a commented license block into its raw text (with trailing newline)."""
    if is_multiline_comment:
        # For multi-line comments, process line by line to preserve formatting
        comment_end = BLOCK_COMMENT_END[comment_start]
        processed_lines: list[str] = []
        for i, line in enumerate(license_block):
            if i == 0:
                # First line: remove the opening delimiter (and the `!` of a /*! comment).
                line = re.sub(rf'^\s*{re.escape(comment_start)}!?\s?', '', line)
                # Skip if first line is now empty (/* was on its own line)
                if line.strip() == '':
                    continue
            elif i == len(license_block) - 1:
                # Last line: remove the closing delimiter.
                line = re.sub(rf'\s?{re.escape(comment_end)}\s*$', '', line)
                # Skip if last line is now empty (*/ was on its own line)
                if line.strip() == '':
                    continue

            # Remove leading * and one space (but preserve additional indentation)
            if line.strip().startswith('*'):
                line = re.sub(r'^\s?\*\s?', '', line)
            elif line.strip() == '':
                # Empty line - keep as is
                line = ''

            processed_lines.append(line)

        return '\n'.join(processed_lines) + '\n'

    # For single-line comments, process line by line
    def uncomment(text: str) -> str:
        return re.sub(rf'^{re.escape(comment_start)}\ ?', '', text)

    return '\n'.join(uncomment(line) for line in license_block) + '\n'


def _skip_first_line_directive(lines: list[str]) -> int:
    """Return the line where the header starts, inserting a blank line after a directive."""
    if not lines or not lines[0].startswith(FIRST_LINE_DIRECTIVE_PREFIXES):
        return 0
    if len(lines) > 1 and lines[1] != '':
        lines.insert(1, '')
    return 2


def _format_license_header(year: str, comment_start: str, is_multiline_comment: bool) -> str:
    """Return a new license header, commented for the file type."""
    license_lines = '\n'.join([make_nvidia_copyright_text(year), license_text])

    if is_multiline_comment:
        body = ''.join(
            f' * {line}\n' if line.strip() else ' *\n' for line in license_lines.splitlines()
        )
        return f'/*\n{body} */'

    commented = textwrap.indent(license_lines, comment_start + ' ', predicate=lambda _: True)
    return '\n'.join(line.rstrip() for line in commented.splitlines())


def _find_apache_notice(words: list[list[str]]) -> tuple[int, int] | None:
    """Return the first and last line of the Apache-2.0 notice paragraph, if present."""
    for start, line_words in enumerate(words):
        if line_words[:3] != APACHE_NOTICE_WORDS[:3]:
            continue
        notice: list[str] = []
        for end in range(start, len(words)):
            notice += words[end]
            if len(notice) >= len(APACHE_NOTICE_WORDS):
                break
        return (start, end) if notice == APACHE_NOTICE_WORDS else None
    return None


def _strip_apache_notice(lines: list[str], dry_run: bool) -> bool:
    """Remove the Apache-2.0 notice paragraph from an NVIDIA-only license block.

    Blocks with any other copyright holder keep their notice, as Apache-2.0 requires.
    """
    if any('Copyright' in line and 'NVIDIA CORPORATION' not in line for line in lines):
        return False
    words = [_COMMENT_MARKER_RE.sub('', line).split() for line in lines]
    notice_range = _find_apache_notice(words)
    if notice_range is None:
        return False
    start, end = notice_range
    # Also drop the empty comment line that separated the notice from the SPDX lines.
    if start > 0 and not words[start - 1]:
        start -= 1
    if not dry_run:
        del lines[start : end + 1]
    return True


def _normalize_license_block(
    filepath: Path, lines: list[str], current_year: int, dry_run: bool
) -> bool:
    """Normalize the NVIDIA copyright lines and drop the Apache-2.0 notice paragraph."""
    # Keep this scoped to the top license block only, to avoid rewriting stray SPDX text later
    # in a file.
    updated_copyright = _normalize_and_update_nvidia_copyrights(
        lines=lines,
        current_year=current_year,
        dry_run=dry_run,
    )
    if updated_copyright:
        logger.info(f'Updating NVIDIA copyright header years/format in {filepath}.')
    stripped_notice = _strip_apache_notice(lines, dry_run)
    if stripped_notice:
        logger.info(f'Removing the Apache-2.0 notice paragraph from {filepath}.')
    return updated_copyright or stripped_notice


def process_file(filepath: Path, dry_run: bool):
    """Process a file to ensure it has a valid license block, or add a new one if it doesn't."""
    resolved_filepath = _resolve_in_repo(filepath)
    lines = resolved_filepath.read_text(encoding='utf-8').splitlines()
    comment_start = get_comment_delimiter(filepath, lines)
    is_multiline_comment = comment_start in BLOCK_COMMENT_END
    file_current_year = _effective_current_year(filepath)

    start_line = _skip_first_line_directive(lines)

    license_block, license_block_start, license_block_end = _extract_license_block(
        lines=lines,
        start_line=start_line,
        comment_start=comment_start,
        is_multiline_comment=is_multiline_comment,
    )

    # Check for noqa directive (adapt for different comment styles)
    noqa_check = 'noqa: license-check'
    if any(noqa_check in line for line in license_block):
        logger.debug(f'Skipping {filepath} because license checking is disabled for this file.')
        return

    updated_nvidia_header = _normalize_license_block(
        filepath=filepath,
        lines=license_block,
        current_year=file_current_year,
        dry_run=dry_run,
    )
    if updated_nvidia_header and not dry_run:
        lines[license_block_start:license_block_end] = license_block

    # Extract license text from comment block
    license_block_text = _license_block_to_text(
        license_block=license_block,
        comment_start=comment_start,
        is_multiline_comment=is_multiline_comment,
    )
    if len(license_block) != 0 and license_regex.match(license_block_text):
        logger.debug(f'Skipping {filepath} because it contains a valid license block.')
        if updated_nvidia_header and not dry_run:
            _write_lines(resolved_filepath, lines)
        return

    logger.info(f'Adding license block to {filepath}.')
    license_header = _format_license_header(
        year=str(file_current_year),
        comment_start=comment_start,
        is_multiline_comment=is_multiline_comment,
    )
    lines.insert(start_line, license_header + '\n')

    if not dry_run:
        _write_lines(resolved_filepath, lines)


def get_comment_delimiter(filepath: Path, lines: list[str] | None = None) -> str:
    """Get the comment delimiter for a file based on its name, extension or shebang."""
    # Check special file name patterns first.
    for pattern_check, pattern_delimiter in SPECIAL_FILE_PATTERNS:
        if pattern_check(filepath.name):
            return pattern_delimiter

    # Check by extension.
    delimiter: str | None = FILE_TYPE_CONFIG.get(filepath.suffix)
    if delimiter is not None:
        return delimiter

    # Extensionless scripts (e.g. ./isaac_os) are identified by their shebang.
    if not filepath.suffix:
        if lines is None:
            lines = filepath.read_text(errors='replace').splitlines()[:1]
        if lines and lines[0].startswith('#!'):
            return '#'

    raise ValueError(f'Unsupported file type: {filepath}')


def is_exempt(filepath: Path) -> bool:
    """Return whether a file intentionally carries no license header."""
    name = filepath.name
    if any(pattern_check(name) for pattern_check, _ in SPECIAL_FILE_PATTERNS):
        return False
    if filepath.suffix in EXEMPT_SUFFIXES or name in EXEMPT_NAMES:
        return True
    # Extensionless dotfiles, e.g. .gitignore or .clang-format.
    if name.startswith('.') and not filepath.suffix:
        return True

    relpath = _repo_relpath(filepath)
    return relpath is not None and any(fnmatch(relpath, p) for p in EXEMPT_PATH_PATTERNS)


def main():
    """Main entry point for the license check script."""
    parser = argparse.ArgumentParser(description='Ensure files have proper license headers')
    parser.add_argument('files', nargs='+', help='Files to process')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without making changes',
    )

    args = parser.parse_args()

    unsupported: list[Path] = []
    for filename in args.files:
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f'File {filename} does not exist')

        if is_exempt(filepath):
            logger.debug(f'Skipping {filepath} because it is exempt from license checking.')
            continue

        try:
            get_comment_delimiter(filepath)
        except ValueError:
            unsupported.append(filepath)
            continue

        process_file(filepath, args.dry_run)

    if unsupported:
        for filepath in unsupported:
            logger.error(
                f'Unsupported file type for license checking: {filepath}. Add its comment style '
                'to FILE_TYPE_CONFIG or exempt it in tools/lint/license_check.py.'
            )
        sys.exit(1)


if __name__ == '__main__':
    main()

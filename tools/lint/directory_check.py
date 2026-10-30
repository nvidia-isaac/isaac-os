#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
A tool to check that directories are using the correct naming convention.

Can be used as a pre-commit hook.
"""

import argparse
import fnmatch
import pathlib
import sys

FORBIDDEN_DIRS = {
    'example': 'examples',
    'test': 'tests',
}
"""Mapping from forbidden directory name to recommended directory name."""

IGNORE_FILE_NAME = '.directory_check_ignore'

ERROR_MESSAGE = """
Error: Directory '{dir_path}' is not allowed because '{dir_name}' is forbidden. Use '{dir_name_allowed}' instead.
       Fix this error with (git mv {dir_path} {dir_path_parent}/{dir_name_allowed})
       Ignore this error with (touch {dir_path}/{ignore_file_name})
"""


def print_stderr(message: str) -> None:
    """Prints a message to stderr."""
    print(message, file=sys.stderr)


def load_gitignore_patterns(gitignore_path: pathlib.Path) -> list[str]:
    """Reads the .gitignore file and returns a list of patterns."""
    patterns: list[str] = []
    if not gitignore_path.is_file():
        return patterns

    with gitignore_path.open('r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                patterns.append(line)
    return patterns


def is_ignored(path: pathlib.Path, gitignore_patterns: list[str]) -> bool:
    """Checks if a folder should be ignored."""
    if (path / IGNORE_FILE_NAME).exists():
        # Is manually ignored.
        return True

    path_str = str(path)
    for pattern in gitignore_patterns:
        if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(path.name, pattern):
            return True

    return False


def check_directories(
    directory_paths: list[pathlib.Path], ignore_patterns: list[str] | None = None
) -> list[pathlib.Path]:
    """Checks that directories are using the correct naming convention.

    Folders from .gitignore are ignored.

    Args:
        directory_paths: The directories to check.
        ignore_patterns: A list of directory name patterns to ignore.

    Returns:
        A list of directories that are not using the correct naming convention.
    """
    wrong_directories: list[pathlib.Path] = []

    all_ignore_patterns = (ignore_patterns or []) + load_gitignore_patterns(
        pathlib.Path.cwd() / '.gitignore'
    )

    for path in directory_paths:
        assert path.is_dir(), f"Path '{path}' is not a directory."

        if path.is_symlink():
            continue

        if is_ignored(path, all_ignore_patterns):
            continue

        if path.name in FORBIDDEN_DIRS:
            print_stderr(
                ERROR_MESSAGE.format(
                    dir_path=path,
                    dir_path_parent=path.parent,
                    dir_name=path.name,
                    dir_name_allowed=FORBIDDEN_DIRS[path.name],
                    ignore_file_name=IGNORE_FILE_NAME,
                )
            )
            wrong_directories.append(path)

    return wrong_directories


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('file_paths', type=pathlib.Path, nargs='+')
    args = parser.parse_args()

    directory_paths = list({path.parent for path in args.file_paths})

    wrong_directories = check_directories(directory_paths)
    return 1 if len(wrong_directories) > 0 else 0


if __name__ == '__main__':
    sys.exit(main())

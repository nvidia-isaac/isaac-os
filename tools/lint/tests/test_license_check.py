# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
import sys
import tempfile

import pytest

from tools.lint import license_check
from tools.lint.license_check import REPO_ROOT
from tools.lint.license_check import is_exempt
from tools.lint.license_check import process_file

YEAR = datetime.now().year
SPDX_LINE = f'SPDX-FileCopyrightText: Copyright (c) {YEAR} NVIDIA CORPORATION & AFFILIATES.'


@pytest.fixture(name='repo_tmp')
def fixture_repo_tmp() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
        yield Path(tmp)


def test_process_file_rejects_file_outside_repository(tmp_path: Path):
    outside_file = tmp_path / 'outside.py'
    original = "print('unchanged')\n"
    outside_file.write_text(original)

    with pytest.raises(ValueError, match='outside the repository'):
        process_file(outside_file, dry_run=False)

    assert outside_file.read_text() == original


@pytest.mark.parametrize(
    ('name', 'body', 'first_line'),
    [
        ('script.sh', 'echo hi\n', '# ' + SPDX_LINE + ' All rights reserved.'),
        ('settings.toml', '[tool]\n', '# ' + SPDX_LINE + ' All rights reserved.'),
        ('entry.ctest', 'include(x)\n', '# ' + SPDX_LINE + ' All rights reserved.'),
        ('Doxyfile', 'GENERATE_XML = YES\n', '# ' + SPDX_LINE + ' All rights reserved.'),
        ('schema.fbs', 'table Foo {}\n', '// ' + SPDX_LINE + ' All rights reserved.'),
        ('index.ts', 'export {};\n', '// ' + SPDX_LINE + ' All rights reserved.'),
        ('config.js', 'module.exports = {};\n', '// ' + SPDX_LINE + ' All rights reserved.'),
        ('kernel.cu', '__global__ void k() {}\n', '/*'),
        ('kernel.cuh', '#pragma once\n', '/*'),
        ('theme.css', 'body {}\n', '/*'),
    ],
)
def test_process_file_adds_header_to_supported_types(
    repo_tmp: Path, name: str, body: str, first_line: str
):
    path = repo_tmp / name
    path.write_text(body)

    process_file(path, dry_run=False)
    with_header = path.read_text()
    assert with_header.splitlines()[0] == first_line
    assert SPDX_LINE in with_header
    assert 'SPDX-License-Identifier: Apache-2.0' in with_header
    assert 'Licensed under the Apache License' not in with_header
    assert with_header.endswith(body)

    # A second pass accepts the header it just wrote.
    process_file(path, dry_run=False)
    assert path.read_text() == with_header


@pytest.mark.parametrize(
    ('name', 'directive'),
    [
        ('layer.usda', '#usda 1.0'),
        ('isaac_tool', '#!/usr/bin/env bash'),
    ],
)
def test_process_file_keeps_directive_first(repo_tmp: Path, name: str, directive: str):
    path = repo_tmp / name
    path.write_text(f'{directive}\nbody\n')

    process_file(path, dry_run=False)

    lines = path.read_text().splitlines()
    assert lines[0] == directive
    assert lines[2] == '# ' + SPDX_LINE + ' All rights reserved.'
    assert lines[-1] == 'body'


@pytest.mark.parametrize(
    ('name', 'header'),
    [
        ('script.sh', '#!/bin/bash\n# {spdx} All rights reserved.\n# {license}\n'),
        ('theme.css', '/*!\n * {spdx} All rights reserved.\n * {license}\n */\n'),
    ],
)
def test_process_file_accepts_short_spdx_header(repo_tmp: Path, name: str, header: str):
    path = repo_tmp / name
    content = header.format(spdx=SPDX_LINE, license='SPDX-License-Identifier: Apache-2.0')
    if content.startswith('#!'):
        content = content.replace('\n', '\n\n', 1)
    content += '\nbody\n'
    path.write_text(content)

    process_file(path, dry_run=False)

    assert path.read_text() == content


@pytest.mark.parametrize(
    ('separator_lines', 'expected_separator_lines'),
    [
        ([], []),
        (
            ['# SPDX-FileCopyrightText: Copyright (c) 2020 Other'],
            ['# SPDX-FileCopyrightText: Copyright (c) 2020 Other'],
        ),
        (['#'], []),
    ],
)
def test_process_file_normalizes_separate_rights_reserved(
    repo_tmp: Path, separator_lines: list[str], expected_separator_lines: list[str]
):
    path = repo_tmp / 'script.sh'
    lines = [
        f'# {SPDX_LINE}',
        *separator_lines,
        '# All rights reserved.',
        '# SPDX-License-Identifier: Apache-2.0',
    ]
    path.write_text('\n'.join(lines) + '\n')

    process_file(path, dry_run=False)

    expected = [
        '# ' + SPDX_LINE + ' All rights reserved.',
        *expected_separator_lines,
        '# SPDX-License-Identifier: Apache-2.0',
    ]
    normalized_lines = path.read_text().splitlines()
    assert normalized_lines == expected
    assert (
        sum(
            'SPDX-FileCopyrightText:' in line and 'NVIDIA CORPORATION' in line
            for line in normalized_lines
        )
        == 1
    )
    assert '# All rights reserved.' not in normalized_lines

    process_file(path, dry_run=False)

    assert path.read_text().splitlines() == normalized_lines


def test_process_file_leaves_inline_rights_reserved_unchanged(repo_tmp: Path):
    path = repo_tmp / 'script.sh'
    content = f'# {SPDX_LINE} All rights reserved.\n# SPDX-License-Identifier: Apache-2.0\n'
    path.write_text(content)

    process_file(path, dry_run=False)

    assert path.read_text() == content


def test_process_file_adds_missing_rights_reserved_inline(repo_tmp: Path):
    path = repo_tmp / 'script.sh'
    path.write_text(f'# {SPDX_LINE}\n# SPDX-License-Identifier: Apache-2.0\n')

    process_file(path, dry_run=False)

    assert path.read_text().splitlines() == [
        '# ' + SPDX_LINE + ' All rights reserved.',
        '# SPDX-License-Identifier: Apache-2.0',
    ]


APACHE_NOTICE = """\
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


def _hash_comment(text: str) -> str:
    return ''.join(f'# {line}'.rstrip() + '\n' for line in text.splitlines())


def _block_comment(text: str) -> str:
    body = ''.join(f' * {line}'.rstrip() + '\n' for line in text.splitlines())
    return f'/*\n{body} */\n'


@pytest.mark.parametrize(
    ('name', 'comment'),
    [
        ('script.py', _hash_comment),
        ('source.cpp', _block_comment),
    ],
)
def test_process_file_strips_apache_notice(repo_tmp: Path, name: str, comment):
    spdx = f'{SPDX_LINE} All rights reserved.\nSPDX-License-Identifier: Apache-2.0\n'
    path = repo_tmp / name
    path.write_text(comment(f'{spdx}\n{APACHE_NOTICE}') + '\nbody\n')

    process_file(path, dry_run=False)

    assert path.read_text() == comment(spdx) + '\nbody\n'


def test_process_file_keeps_third_party_apache_notice(repo_tmp: Path):
    path = repo_tmp / 'vendored.py'
    content = (
        _hash_comment(
            f'{SPDX_LINE}\nAll rights reserved.\nSPDX-License-Identifier: Apache-2.0\n\n'
            f'Copyright 2020 Other Authors\n\n{APACHE_NOTICE}'
        )
        + '\nbody\n'
    )
    path.write_text(content)

    process_file(path, dry_run=False)

    expected = (
        _hash_comment(
            f'{SPDX_LINE} All rights reserved.\nSPDX-License-Identifier: Apache-2.0\n\n'
            f'Copyright 2020 Other Authors\n\n{APACHE_NOTICE}'
        )
        + '\nbody\n'
    )
    assert path.read_text() == expected


@pytest.mark.parametrize(
    'relpath',
    [
        '.gitignore',
        '.clang-format',
        'docs/.dockerignore',
        'LICENSE.txt',
        'docs/requirements.txt',
        'README.md',
        'docs/overview.mdx',
        'docs/api/cpp/index.rst',
        'metadata.json',
        'fix.patch',
        'logo.svg',
        'model.onnx',
        'CODE_OF_CONDUCT',
        'docs/fern/dist/output.js',
        'tools/foxglove/isaac-to-foxglove-converter/src/generated/holoscan-schema/x.ts',
    ],
)
def test_is_exempt(relpath: str):
    assert is_exempt(REPO_ROOT / relpath)


@pytest.mark.parametrize(
    'relpath',
    ['CMakeLists.txt', 'Dockerfile', 'docs/Doxyfile', 'isaac_os', 'tools/lint/run_lint.sh'],
)
def test_is_not_exempt(relpath: str):
    assert not is_exempt(REPO_ROOT / relpath)


def test_main_fails_on_unsupported_file_type(repo_tmp: Path, monkeypatch: pytest.MonkeyPatch):
    unsupported = repo_tmp / 'data.xyz'
    unsupported.write_text('payload\n')
    supported = repo_tmp / 'script.sh'
    supported.write_text('echo hi\n')
    monkeypatch.setattr(sys, 'argv', ['license_check.py', str(unsupported), str(supported)])

    with pytest.raises(SystemExit) as exc_info:
        license_check.main()

    assert exc_info.value.code == 1
    assert unsupported.read_text() == 'payload\n'
    assert SPDX_LINE in supported.read_text()

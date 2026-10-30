#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a disposable documentation tree with explicit source-browser links."""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from docs.scripts.check_doc_links import check  # pylint: disable=wrong-import-position
from docs.scripts.check_doc_links import load_page_routes  # pylint: disable=wrong-import-position
from docs.scripts.check_doc_links import prose_mask  # pylint: disable=wrong-import-position
from docs.scripts.check_doc_links import replace_target  # pylint: disable=wrong-import-position
from docs.scripts.check_doc_links import rewrite_links  # pylint: disable=wrong-import-position
from docs.scripts.sync_application_pages import (  # pylint: disable=wrong-import-position
    SourceContext,
)
from docs.scripts.sync_application_pages import (  # pylint: disable=wrong-import-position
    discover_pages,
)
from docs.scripts.sync_application_pages import render  # pylint: disable=wrong-import-position
from docs.scripts.sync_application_pages import (  # pylint: disable=wrong-import-position
    rewrite_source_links,
)

PUBLIC_REPOSITORY = 'https://github.com/nvidia-isaac/isaac-os'
RELEASE_TAG = '<release_tag>'


def resolve_prompt_links(content: str, site_url: str) -> str:
    """Make copied prompt guide links absolute Markdown URLs."""
    site = urlsplit(site_url)

    def resolve(match, asset):
        url = urlsplit(match['target'].strip('<>'))
        if asset or url.scheme or url.netloc or not url.path.startswith('/'):
            return match.group()
        path = url.path.rstrip('/')
        if not Path(path).suffix:
            path += '.md'
        return replace_target(
            match, urlunsplit((site.scheme, site.netloc, path, url.query, url.fragment))
        )

    parts: list[str] = []
    end = 0
    for prompt in re.finditer(
        r'<Prompt\b[^>]*(?<!/)>(.*?)</Prompt\s*>', prose_mask(content), re.DOTALL
    ):
        start, stop = prompt.span(1)
        parts.extend((content[end:start], rewrite_links(content[start:stop], resolve)))
        end = stop
    return ''.join(parts) + content[end:]


def archive_checkout(checkout: Path, ref: str, output: Path) -> None:
    """Extract tracked files from one ref without Git metadata or external links."""
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Git snapshot output must be empty.')
    with tempfile.TemporaryFile() as archive:
        subprocess.run(['git', 'archive', ref], cwd=checkout, stdout=archive, check=True)
        archive.seek(0)
        with tarfile.open(fileobj=archive) as source:
            members = source.getmembers()
            if any(
                not (member.isfile() or member.isdir())
                or Path(member.name).is_absolute()
                or '..' in Path(member.name).parts
                for member in members
            ):
                raise ValueError('Git snapshot contains an unsafe path or symbolic link.')
            # Member checks also protect Python versions without extraction filters.
            if hasattr(tarfile, 'data_filter'):
                source.extractall(output, members=members, filter='data')
            else:
                source.extractall(output, members=members)


def prepare(root: Path, context: SourceContext) -> None:
    """Resolve a disposable snapshot in place using this tooling checkout."""
    root = root.resolve()
    pages = discover_pages(root)
    rendered = {
        page: render(source, root, pages, context=context) for source, page in pages.items()
    }
    for page in load_page_routes(root / 'docs/fern'):
        if page not in rendered and page.is_file():
            rendered[page] = rewrite_source_links(page.read_text(), page, root, pages, context)
    # Previews keep the placeholder: their refs may not exist in the public repository.
    if context.repository == PUBLIC_REPOSITORY:
        rendered = {
            page: content.replace(RELEASE_TAG, context.ref) for page, content in rendered.items()
        }
    for page, content in rendered.items():
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(content)
    findings = check(root / 'docs', root / 'docs/fern', published=True)
    if findings:
        raise ValueError('\n'.join(str(finding) for finding in findings))


def prepare_copy(root: Path, output: Path, context: SourceContext) -> None:
    """Copy content, including local edits for previews, without Git metadata or ignored output."""
    root, output = root.resolve(), output.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Publication output must be separate from its source checkout.')
    if (root / '.git').exists():
        paths = (
            subprocess.check_output(
                ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=root
            )
            .decode()
            .split('\0')
        )
        output.mkdir()
        for name in filter(None, paths):
            source, target = root / name, output / name
            if not source.exists() and not source.is_symlink():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target, follow_symlinks=False)
    else:
        shutil.copytree(root, output, symlinks=True)
    prepare(output, context)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-repository', required=True)
    parser.add_argument('--source-ref', required=True)
    args = parser.parse_args()
    try:
        prepare_copy(args.root, args.output, SourceContext(args.source_repository, args.source_ref))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

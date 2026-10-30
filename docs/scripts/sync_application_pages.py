#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate Fern pages from application README.md, BENCHMARK.md and RESULTS.md files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from difflib import unified_diff
import os
from pathlib import Path
import re
import sys
from urllib.parse import quote
from urllib.parse import unquote
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

# Direct script execution puts docs/scripts on sys.path; add the repository
# root so CLI and module callers use the same package import.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from docs.scripts.check_doc_links import load_page_routes  # pylint: disable=wrong-import-position
from docs.scripts.check_doc_links import prose_mask  # pylint: disable=wrong-import-position
from docs.scripts.check_doc_links import replace_target  # pylint: disable=wrong-import-position
from docs.scripts.check_doc_links import rewrite_links  # pylint: disable=wrong-import-position

REPO_ROOT = Path(__file__).resolve().parents[2]
README = 'README.md'
LICENSE = """{/*
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}

"""
GENERATED = (
    '{/* Generated from an application README by docs/scripts/sync_application_pages.py. */}\n\n'
)
# Match the image label first; scan its destination separately to balance parentheses.
IMAGE = re.compile(r'!\[((?:[^\]\[]|\[[^\]]*\])*)\]\(\s*')
IMAGE_END = re.compile(r"""(\s+(?:"[^"\n]*"|'[^'\n]*'|\([^\)\n]*\)))?\s*\)""")


@dataclass(frozen=True)
class SourceContext:
    """Source-browser identity for one documentation snapshot."""

    repository: str
    ref: str

    def __post_init__(self):
        url = urlsplit(self.repository)
        if (
            url.scheme != 'https'
            or not url.hostname
            or url.username
            or url.password
            or not url.path.strip('/')
            or url.query
            or url.fragment
        ):
            raise ValueError('Use an HTTPS repository URL without credentials, query or fragment.')
        if not self.ref or any(part in ('', '.', '..') for part in self.ref.split('/')):
            raise ValueError('A source branch or tag is required.')

    def link(self, path: Path, *, directory: bool = False) -> str:
        separator = '/-' if urlsplit(self.repository).netloc != 'github.com' else ''
        kind = 'tree' if directory else 'blob'
        repository = self.repository.rstrip('/')
        ref = quote(self.ref, safe='/')
        source_path = '' if path == Path('.') else quote(path.as_posix(), safe='/')
        return f'{repository}{separator}/{kind}/{ref}/{source_path}'


def image_destination(content: str, start: int) -> tuple[int, str, str] | None:
    """Read an inline image destination and title, including balanced parentheses."""
    if start >= len(content):
        return None
    end = start
    bracketed = content[start] == '<'
    if bracketed:
        end = content.find('>', start + 1)
        if end == -1:
            return None
        target = content[start + 1 : end]
        end += 1
    else:
        depth = 0
        while end < len(content):
            char = content[end]
            if char == '\\' and end + 1 < len(content):
                end += 2
                continue
            if char.isspace() or (char == ')' and depth == 0):
                break
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            end += 1
        if depth:
            return None
        target = content[start:end]
    ending = IMAGE_END.match(content, end)
    if not target or ending is None:
        return None
    # Markdown backslash escapes are syntax, not part of the filesystem path.
    target = re.sub(r'\\([!"#$%&\'()*+,\-./:;<=>?@\[\]\\^_`{|}~])', r'\1', target)
    return ending.end(), target, ending.group(1) or ''


def rewrite_images(content: str, source: Path, root: Path, page: Path | None) -> str:
    """Relocate relative images to the generated page, preserving Markdown syntax."""

    def replace_image(label: str, target: str, title: str, original: str) -> str:
        url = urlsplit(target)
        if url.scheme or not url.path or url.path.startswith('/'):
            return original
        path = (source.parent / unquote(url.path)).resolve()
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError(f'{source}: missing or out-of-tree image: {target}')
        if page is None:
            raise ValueError(f'{source}: image rewriting requires a generated page path')
        # Encode spaces and parentheses so the relocated destination is unambiguous
        # to both Markdown renderers and the documentation link checker.
        relative = Path(os.path.relpath(path, page.parent)).as_posix()
        destination = urlunsplit(('', '', quote(relative, safe='/'), url.query, url.fragment))
        return f'![{label}]({destination}{title})'

    parts = []
    cursor = 0
    mask = prose_mask(content)
    for match in IMAGE.finditer(content):
        if match.start() < cursor or not mask[match.start()].strip():
            continue
        parsed = image_destination(content, match.end())
        if parsed is None:
            continue
        end, target, title = parsed
        parts.append(content[cursor : match.start()])
        parts.append(replace_image(match.group(1), target, title, content[match.start() : end]))
        cursor = end
    parts.append(content[cursor:])
    return ''.join(parts)


def source_destination(target, source, root, routes, context):
    """Return a guide/source URL, or None for a source link without publication context."""
    original = target
    target = target.removeprefix('<').removesuffix('>')
    url = urlsplit(target)
    if url.scheme or not url.path or url.path.startswith('/'):
        return original
    path = (source.parent / unquote(url.path)).resolve()
    if not path.exists() or not path.is_relative_to(root):
        raise ValueError(f'{source}: missing or out-of-tree link target: {target}')
    if route := routes.get(path):
        return urlunsplit(('', '', route, url.query, url.fragment))
    if context:
        return context.link(path.relative_to(root), directory=path.is_dir()) + urlunsplit(
            ('', '', '', url.query, url.fragment)
        )
    return None


def rewrite_source_links(
    content: str,
    source: Path,
    root: Path,
    pages: dict[Path, Path] | None = None,
    context: SourceContext | None = None,
) -> str:
    """Resolve relative page links and source links against the selected content tree."""
    pages = pages or {}
    routes = load_page_routes(root / 'docs' / 'fern')
    for readme, page in pages.items():
        if page not in routes:
            raise ValueError(f'generated page is missing from Fern navigation: {page}')
        routes[readme] = routes[page]

    def replace_link(match, asset):
        if asset:
            return match.group()
        target = match['target'].strip('<>')
        resolved = source_destination(target, source, root, routes, context)
        if resolved is None:
            url = urlsplit(target)
            path = (source.parent / unquote(url.path)).resolve()
            if 'label' in match.groupdict():
                label = match['label']
                return f'{label} (`{path.relative_to(root).as_posix()}`)'
            page = pages.get(source, source)
            resolved = urlunsplit(
                (
                    '',
                    '',
                    quote(os.path.relpath(path, page.parent), safe='/'),
                    url.query,
                    url.fragment,
                )
            )
        return replace_target(match, resolved)

    return rewrite_links(content, replace_link)


def render(
    source: Path,
    root: Path,
    pages: dict[Path, Path] | None = None,
    *,
    context: SourceContext | None = None,
) -> str:
    """Convert one Markdown source to MDX with optional source-browser links."""
    content = source.read_text(encoding='utf-8')

    content = rewrite_source_links(content, source, root, pages, context)
    content = rewrite_images(content, source, root, (pages or {}).get(source))
    source_label = 'README' if source.name == README else source.name
    return LICENSE + GENERATED.replace('README', source_label) + content


def is_excluded_application_directory(relative: Path) -> bool:
    """Exclude the applications root and the internal benchmark scripts subtree."""
    return relative == Path('.') or relative.is_relative_to('robot_control_benchmark/scripts')


def discover_pages(root: Path) -> dict[Path, Path]:
    """Map application Markdown sources to unique generated paths."""
    pages: dict[Path, Path] = {}
    targets: set[Path] = set()
    applications = root / 'applications'
    for name in (README, 'BENCHMARK.md', 'RESULTS.md'):
        for source in sorted(applications.rglob(name)):
            if not source.resolve().is_relative_to(applications):
                raise ValueError(f'out-of-tree application Markdown: {source}')
            relative = source.parent.relative_to(applications)
            if is_excluded_application_directory(relative):
                continue
            if name != README:
                relative /= source.stem.lower()
            target = root / 'docs' / 'applications' / f'{relative.as_posix()}.mdx'
            if target in targets:
                raise ValueError(f'multiple sources generate {target}')
            pages[source] = target
            targets.add(target)
    return pages


def sync_pages(root: Path, *, check_only: bool = False) -> int:
    """Generate or check application overview, benchmark and result pages.

    Preserve the source directory structure: applications/<path>/README.md
    becomes docs/applications/<path>.mdx, including nested application guides.
    BENCHMARK.md and RESULTS.md become <path>/benchmark.mdx and <path>/results.mdx.
    Sidebar titles, ordering, and public slugs remain in Fern navigation.
    """
    root = root.resolve()
    try:
        pages = discover_pages(root)
        expected_targets = set(pages.values())
        orphans = [
            path.relative_to(root).as_posix()
            for path in sorted((root / 'docs/applications').rglob('*.mdx'))
            if path not in expected_targets
            and '{/* Generated from an application ' in path.read_text(encoding='utf-8')
        ]
        if orphans:
            raise ValueError(
                'Generated application pages have no source Markdown: '
                + ', '.join(orphans)
                + '. Remove these pages and update docs/fern/index.yml.'
            )
        # Render every page before writing, so invalid links or navigation cannot leave
        # a partially regenerated set of pages.
        rendered = {source: render(source, root, pages) for source in pages}
    except (ValueError, OSError) as error:
        print(error, file=sys.stderr)
        return 1

    stale = False
    for source, target in pages.items():
        expected = rendered[source]
        actual = target.read_text(encoding='utf-8') if target.exists() else ''
        if actual == expected:
            continue
        stale = True
        if check_only:
            print(
                ''.join(
                    unified_diff(
                        actual.splitlines(keepends=True),
                        expected.splitlines(keepends=True),
                        fromfile=str(target),
                        tofile=f'{source} (rendered)',
                    )
                ),
                end='',
            )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(expected, encoding='utf-8')
            print(f'Updated {target.relative_to(root)}')

    if stale and check_only:
        print(
            'Generated application pages are stale; run '
            'python3 docs/scripts/sync_application_pages.py'
        )
        return 1
    if check_only:
        print('Generated pages match all application Markdown sources')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='fail if committed pages differ')
    args = parser.parse_args()
    return sync_pages(REPO_ROOT, check_only=args.check)


if __name__ == '__main__':
    raise SystemExit(main())

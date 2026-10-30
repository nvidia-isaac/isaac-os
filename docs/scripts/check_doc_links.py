#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Validate links in the Isaac OS Markdown and Fern user guide.

``fern check --local`` validates configuration and MDX syntax, but not the targets of
internal links. A link written as a bare relative path (``guides/isaac-sim``)
is published verbatim, so the browser resolves it against the current page's directory
and lands on a path that does not exist. With ``hide-404-page: true`` in ``docs.yml``
those requests are redirected to the docs homepage, which looks like a working link.

Inter-page links must therefore be published site paths starting with ``/``
(``/isaac-os/...``), with ``#anchor`` alone for same-page links. Fern also
rewrites paths to committed page files (``sdk_installation.mdx``), but its documentation
lists that form as unsupported, so a CLI upgrade could publish those verbatim and break
them; this script reports them so the site paths stay canonical.

This script derives every published page path from the Fern navigation, collects the
anchors each page defines, and reports links that cannot resolve:

* relative link targets that Fern publishes unchanged,
* page-file (``.mdx``) paths that should be site paths,
* page paths with no matching navigation entry,
* ``#anchor`` fragments no heading, ``<Anchor id>``, or component title provides,
* image or asset paths missing from disk,
* ``docs.yml`` redirect destinations that do not resolve.

Adapted from Holoscan SDK's ``public/docs/scripts/check_doc_links.py`` at
``79dc6cb48e``. Checks the authored pages and version snapshots used by Isaac OS.

Usage (from the Isaac OS repository root):

  python3 docs/scripts/check_doc_links.py
  python3 docs/scripts/check_doc_links.py --docs-root docs
  python3 docs/scripts/check_doc_links.py --local-files-only
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
from urllib.parse import unquote
from urllib.parse import urlsplit

import yaml

SCRIPTS = Path(__file__).resolve().parent
SCRIPT_DOCS_ROOT = SCRIPTS.parent
DEFAULT_SITE_PREFIX = '/isaac-os'

# Fern derives anchors from these component titles in addition to headings.
_TITLED_COMPONENTS = ('Tab', 'Accordion', 'AccordionGroup', 'Step', 'Card')

# Each form names its destination so checking and rewriting use the same spans.
_LINK_RE = re.compile(
    r'(?P<image>!?)\[(?P<label>(?:[^\]\[]|\[[^\]]*\])*)\]\(\s*'
    r'(?P<target><[^>]+>|[^)\s]+)'
    r"""(?:\s+(?:"[^"\n]*"|'[^'\n]*'|\([^\)\n]*\)))?\s*\)"""
)
_REFERENCE_LINK_RE = re.compile(
    r'^\s*\[(?!\^)[^]\n]+\]:\s*(?P<target><[^>\n]+>|[^\s]+)', re.MULTILINE
)
_HTML_TARGET_RE = re.compile(
    r"""\b(?P<attribute>href|src)=(?P<quote>["'])(?P<target>.*?)(?P=quote)"""
)

# Capture heading text for anchor generation; callers match at the line start.
_HEADING_RE = re.compile(r'#{1,6}\s+(.*)')
_FENCE_RE = re.compile(r'\s*(`{3,}|~{3,})')
_ASSET_RE = re.compile(r'\.(png|jpe?g|gif|svg|webp|pdf)$', re.IGNORECASE)
_EXTERNAL_RE = re.compile(r'^(?:[A-Za-z][A-Za-z0-9+.-]*:|//)')
_PARENT_PREFIX_RE = re.compile(r'^(?:\.\./)+')


class Finding:
    """A single link problem, rendered as one ``file:line: message`` report line."""

    def __init__(self, path: Path, line: int, message: str) -> None:
        self.path = path
        self.line = line
        self.message = message

    def __str__(self) -> str:
        return f'{self.path}:{self.line}: {self.message}'


def _slug_from_title(title: str) -> str:
    """Slugify a navigation title the way Fern does, splitting camelCase runs.

    ``Use iGPU with dGPU`` becomes ``use-i-gpu-with-d-gpu`` and ``Enabling GPUDirect
    RDMA`` becomes ``enabling-gpu-direct-rdma``.
    """
    spaced = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', title)
    spaced = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', spaced)
    return re.sub(r'-{2,}', '-', re.sub(r'[^A-Za-z0-9]+', '-', spaced).strip('-').lower())


def _slug_from_heading(text: str) -> str:
    """Slugify heading or component-title text the way Fern does.

    Unlike navigation titles, heading anchors keep camelCase runs together
    (``BlockMemoryPool`` becomes ``blockmemorypool``) and preserve underscores.
    """
    plain = re.sub(r'`([^`]*)`', r'\1', text)
    plain = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', plain)
    plain = re.sub(r'<[^>]+>', '', plain)
    plain = re.sub(r'\*+', '', plain)
    plain = re.sub(r'[^A-Za-z0-9\s_-]+', '', plain)
    return re.sub(r'[\s-]+', '-', plain).strip('-').lower()


def _strip_code(text: str) -> list[str]:
    """Blank out fenced code blocks, keeping line numbers aligned with the source."""
    lines: list[str] = []
    fence: str | None = None
    for line in text.split('\n'):
        marker = _FENCE_RE.match(line)
        if fence is None and marker:
            fence = marker.group(1)
            lines.append('')
        elif fence is not None:
            if marker:
                candidate = marker.group(1)
                if (
                    candidate[0] == fence[0]
                    and len(candidate) >= len(fence)
                    and not line[marker.end() :].strip()
                ):
                    fence = None
            lines.append('')
        else:
            lines.append(line)
    return lines


def prose_mask(text: str) -> str:
    """Mask fenced and inline code without changing offsets or line numbers."""
    mask = '\n'.join(
        visible if visible else ' ' * len(original)
        for original, visible in zip(text.split('\n'), _strip_code(text))
    )
    parts = []
    end = 0
    for start, stop in _inline_code_spans(mask):
        parts.append(mask[end:start])
        parts.append(re.sub(r'[^\n]', ' ', mask[start:stop]))
        end = stop
    parts.append(mask[end:])
    return ''.join(parts)


def _inline_code_spans(text):
    # Index matching backtick runs once instead of rescanning unmatched suffixes.
    tokens = list(re.finditer(r'`+|\n[ \t]*\n', text))
    next_run: dict[int, int] = {}
    closing = {}
    for index in range(len(tokens) - 1, -1, -1):
        token = tokens[index]
        if text[token.start()] == '\n':
            next_run.clear()
            continue
        length = token.end() - token.start()
        if length in next_run:
            closing[index] = next_run[length]
        next_run[length] = token.end()

    end = 0
    for index, token in enumerate(tokens):
        start = token.start()
        if start < end or (start and text[start - 1] == '\\'):
            continue
        if index in closing:
            end = closing[index]
            yield start, end


def read_yaml(path: Path) -> dict:
    """Read a Fern configuration mapping."""
    try:
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
    except yaml.YAMLError as error:
        raise ValueError(f'{path}: {error}') from error
    if not isinstance(config, dict):
        raise ValueError(f'{path}: expected a YAML mapping')
    return config


def _load_nav(path: Path) -> list[dict]:
    return read_yaml(path).get('navigation', [])


def _load_redirects(docs_yml: Path) -> list[tuple[str, str]]:
    return [
        (item['source'], item['destination']) for item in read_yaml(docs_yml).get('redirects', [])
    ]


def _site_prefix(docs_yml: Path) -> str:
    instances = read_yaml(docs_yml).get('instances', [])
    if not instances:
        return DEFAULT_SITE_PREFIX
    instance = instances[0]
    address = instance.get('custom-domain') or instance['url']
    return urlsplit(address if '://' in address else 'https://' + address).path.rstrip('/')


def _walk_nav(
    nodes: list[dict],
    prefix: str,
    fern_dir: Path,
    pages: dict[Path, str],
) -> None:
    for node in nodes:
        if 'section' in node:
            section_path = prefix
            if not node.get('skip-slug', False):
                slug = node.get('slug') or _slug_from_title(node['section'])
                section_path = f'{prefix}/{slug}'
            if node.get('path'):
                pages[(fern_dir / node['path']).resolve()] = section_path
            _walk_nav(
                node.get('contents', []),
                section_path,
                fern_dir,
                pages,
            )
        elif 'page' in node and node.get('path'):
            slug = node.get('slug') or _slug_from_title(node['page'])
            pages[(fern_dir / node['path']).resolve()] = f'{prefix}/{slug}'


def load_page_routes(fern_dir: Path) -> dict[Path, str]:
    """Read published page routes from Fern configuration."""
    nav_file = fern_dir / 'index.yml'
    docs_yml = fern_dir / 'docs.yml'
    for path in (nav_file, docs_yml):
        if not path.is_file():
            raise ValueError(f'missing Fern configuration: {path}')
    pages: dict[Path, str] = {}
    prefix = _site_prefix(docs_yml)
    _walk_nav(_load_nav(nav_file), prefix, fern_dir, pages)
    if not pages:
        raise ValueError(f'no pages found in {nav_file}')
    return pages


def _collect_anchors(lines: list[str]) -> set[str]:
    """Collect every fragment the page provides: headings, anchors, and tab titles."""
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    for line in lines:
        heading = _HEADING_RE.match(line)
        if heading:
            base = _slug_from_heading(heading.group(1))
            if base:
                # Fern disambiguates repeated headings with a numeric suffix.
                count = seen.get(base, 0)
                seen[base] = count + 1
                anchors.add(base if count == 0 else f'{base}-{count}')
        anchors.update(re.findall(r'\bid="([A-Za-z0-9][A-Za-z0-9_-]*)"', line))
        for component in _TITLED_COMPONENTS:
            for title in re.findall(rf'<{component}\b[^>]*\btitle="([^"]+)"', line):
                anchors.add(_slug_from_heading(title))
    return anchors


def link_matches(text: str):
    """Yield link matches and asset flags outside code, in source order."""
    mask = prose_mask(text)
    matches = sorted(
        (
            match
            for pattern in (_LINK_RE, _REFERENCE_LINK_RE, _HTML_TARGET_RE)
            for match in pattern.finditer(text)
        ),
        key=lambda match: match.start(),
    )
    end = 0
    for match in matches:
        start = match.start() + len(match.group()) - len(match.group().lstrip())
        if match.start() < end or not mask[start].strip():
            continue
        end = match.end()
        fields = match.groupdict()
        yield match, bool(fields.get('image') or fields.get('attribute') == 'src')


def replace_target(match, target: str) -> str:
    start, end = match.span('target')
    if match['target'].startswith('<'):
        target = '<' + target + '>'
    return match.group()[: start - match.start()] + target + match.group()[end - match.start() :]


def rewrite_links(text: str, replace) -> str:
    """Replace recognized links without touching surrounding prose or code examples."""
    parts: list[str] = []
    cursor = 0
    for match, asset in link_matches(text):
        parts.extend((text[cursor : match.start()], replace(match, asset)))
        cursor = match.end()
    return ''.join(parts) + text[cursor:]


def _link_targets(lines: list[str]) -> list[tuple[int, str]]:
    text = '\n'.join(lines)
    return [
        (text.count('\n', 0, match.start()) + 1, match['target'].strip('<>'))
        for match, _ in link_matches(text)
    ]


def _markdown_sources(public_root: Path) -> list[Path]:
    """Return authored Markdown inputs from the repository tree."""
    excluded_parts = {
        '.cache',
        '.copybara',
        '.local',
        'dist',
        'generated',
        'node_modules',
        'skills',
    }
    sources = []
    for suffix in ('*.md', '*.mdx'):
        for path in public_root.rglob(suffix):
            relative = path.relative_to(public_root)
            if any(part in excluded_parts for part in relative.parts):
                continue
            if any(
                re.fullmatch(r'(?:build|install|gxf-build)(?:-.*)?', part)
                for part in relative.parts
            ):
                continue
            sources.append(path.resolve())
    return sorted(set(sources))


def _check_local_target(target: str, source: Path, public_root: Path) -> str | None:
    page = unquote(re.split(r'[?#]', target, maxsplit=1)[0])
    if source.suffix == '.md' and (
        page == DEFAULT_SITE_PREFIX or page.startswith(DEFAULT_SITE_PREFIX + '/')
    ):
        return f'guide route in Markdown: use a relative source path instead of {target}'
    if (
        not target
        or _EXTERNAL_RE.match(target)
        or target.startswith(('#', '?', DEFAULT_SITE_PREFIX))
        or any(character in target for character in '{}*$')
    ):
        return None
    if not page:
        return None
    candidate = (
        public_root / page.lstrip('/') if page.startswith('/') else source.parent / page
    ).resolve()
    if not candidate.is_relative_to(public_root):
        return f'local link escapes the repository tree: {target}'
    if not candidate.exists():
        return f'missing local target: {target}'
    return None


def check_local_file_links(public_root: Path) -> list[Finding]:
    """Check local targets and reject published guide routes in Markdown.

    MDX guide routes are handled by :func:`check`; this pass checks paths that
    GitHub and local Markdown renderers resolve as files, including targets moved or
    deleted elsewhere in the repository.
    """
    public_root = public_root.resolve()
    sources = _markdown_sources(public_root)
    findings: list[Finding] = []

    for source in sources:
        relative = source.relative_to(public_root)
        lines = _strip_code(source.read_text(encoding='utf-8'))
        for lineno, target in _link_targets(lines):
            message = _check_local_target(target.strip(), source, public_root)
            if message:
                findings.append(Finding(relative, lineno, message))

    print(f'Checked local file targets in {len(sources)} Markdown files.')
    return findings


def _check_relative_target(page, anchor, target, path, *, pages, anchors, prefix, nav_file):
    if page.endswith(('.mdx', '.md')):
        resolved = pages.get((path.parent / page).resolve())
        if resolved is None:
            return f'link to a file outside the navigation: {target}'
        # Fern calls page-file links unsupported, even though it currently rewrites them.
        suggestion = f'{resolved}#{anchor}' if anchor else resolved
        return f'page-file link: use the site path {suggestion}'
    relative_page = _PARENT_PREFIX_RE.sub('', page).rstrip('/')
    candidate = f'{prefix}/{relative_page}'
    if candidate in anchors:
        fragment = '#' + anchor if anchor else ''
        hint = f'use {candidate}{fragment}'
    else:
        hint = f'no page at {candidate}; check the slugs in {nav_file.name}'
    return f'relative link is published unchanged: {target} ({hint})'


def _check_site_target(page, anchor, anchors, redirect_sources):
    resolved = page.rstrip('/')
    if resolved in redirect_sources:
        return None
    if resolved not in anchors:
        return f'no such page: {resolved}'
    if anchor and anchor not in anchors[resolved]:
        return f"no anchor '{anchor}' on {resolved}"
    return None


def _check_target(
    target, path, *, pages, anchors, redirect_sources, root, published, prefix, nav_file
):
    if _EXTERNAL_RE.match(target) or target.startswith('?'):
        return None
    page, _, anchor = target.partition('#')
    page = page.partition('?')[0]
    if _ASSET_RE.search(page):
        if page and not (path.parent / unquote(page)).resolve().is_file():
            return f'missing asset: {target}'
        return None
    if not page:
        own = pages.get(path)
        if anchor and own and anchor not in anchors.get(own, set()):
            return f"no anchor '{anchor}' on this page"
        return None
    return _check_page_target(
        page,
        anchor,
        target,
        path,
        pages=pages,
        anchors=anchors,
        redirect_sources=redirect_sources,
        root=root,
        published=published,
        prefix=prefix,
        nav_file=nav_file,
    )


def _check_page_target(
    page,
    anchor,
    target,
    path,
    *,
    pages,
    anchors,
    redirect_sources,
    root,
    published,
    prefix,
    nav_file,
):
    if not page.startswith('/'):
        candidate = (path.parent / unquote(page)).resolve()
        if candidate.exists() and candidate.is_relative_to(root) and candidate not in pages:
            return f'unresolved source link: {target}' if published else None
    if page.endswith(('.mdx', '.md')) or not page.startswith('/'):
        return _check_relative_target(
            page,
            anchor,
            target,
            path,
            pages=pages,
            anchors=anchors,
            prefix=prefix,
            nav_file=nav_file,
        )
    return _check_site_target(page, anchor, anchors, redirect_sources)


def _check_redirects(redirects, redirects_file, report_root, anchors):
    findings = []
    for source, destination in redirects:
        if destination.startswith('http'):
            continue
        page = destination.split('#')[0].rstrip('/')
        if re.search(r'/:[A-Za-z]\w*\*?(?:/|$)', page, flags=re.ASCII):
            continue
        if page not in anchors:
            findings.append(
                Finding(
                    redirects_file.resolve().relative_to(report_root.resolve()),
                    0,
                    f'redirect {source} -> unknown page {destination}',
                )
            )
    return findings


def _check_navigation_pages(pages, nav_label):
    findings = []
    routes: set[str] = set()
    for page_path, route in pages.items():
        if not page_path.is_file():
            findings.append(Finding(nav_label, 0, f'missing navigation page: {page_path}'))
        if route in routes:
            findings.append(Finding(nav_label, 0, f'duplicate navigation route: {route}'))
        routes.add(route)
    return findings


def _check_navigation(
    *,
    source_root: Path,
    nav_file: Path,
    nav_root: Path,
    prefix: str,
    report_root: Path,
    redirects_file: Path | None = None,
    published: bool = False,
    extra_anchors: dict[str, set[str]] | None = None,
    repository_root: Path | None = None,
) -> list[Finding]:
    """Validate one version of the guide."""
    if not nav_file.is_file():
        raise ValueError(f'missing Fern navigation: {nav_file}')

    pages: dict[Path, str] = {}
    _walk_nav(_load_nav(nav_file), prefix, nav_root, pages)
    if not pages:
        raise ValueError(f'no pages found in {nav_file}')
    findings: list[Finding] = []
    findings.extend(
        _check_navigation_pages(
            pages,
            nav_file.resolve().relative_to(report_root.resolve()),
        )
    )
    redirects = _load_redirects(redirects_file) if redirects_file is not None else []
    redirect_sources = {
        source.rstrip('/')
        for source, _ in redirects
        if not re.search(r'/:[A-Za-z]\w*\*?(?:/|$)', source, flags=re.ASCII)
    }

    sources = sorted(
        path.resolve()
        for path in source_root.rglob('*.mdx')
        if 'generated' not in path.relative_to(source_root).parts
    )
    body_by_path = {path: _strip_code(path.read_text(encoding='utf-8')) for path in sources}
    anchors_by_site_path = {
        site_path: _collect_anchors(body_by_path[path])
        for path, site_path in pages.items()
        if path in body_by_path
    }
    anchors_by_site_path = (extra_anchors or {}) | anchors_by_site_path

    for path in sources:
        rel = path.relative_to(report_root.resolve())
        for lineno, target in _link_targets(body_by_path[path]):
            message = _check_target(
                target,
                path,
                pages=pages,
                anchors=anchors_by_site_path,
                redirect_sources=redirect_sources,
                root=repository_root or source_root.parent,
                published=published,
                prefix=prefix,
                nav_file=nav_file,
            )
            if message:
                findings.append(Finding(rel, lineno, message))

    if redirects_file is not None:
        findings.extend(
            _check_redirects(redirects, redirects_file, report_root, anchors_by_site_path)
        )

    print(
        f'Checked {len(sources)} pages against {len(anchors_by_site_path)} '
        f'navigation entries from {nav_file.relative_to(report_root)}.'
    )
    return findings


def check(
    docs_root: Path,
    fern_dir: Path,
    *,
    published: bool = False,
) -> list[Finding]:
    docs_yml = fern_dir / 'docs.yml'
    if not docs_yml.is_file():
        raise ValueError(f'missing Fern config: {docs_yml}')

    prefix = _site_prefix(docs_yml)
    versions = read_yaml(docs_yml).get('versions') or [{'path': './index.yml'}]
    navigation = [(fern_dir / version['path']).resolve() for version in versions]
    report_root = Path(os.path.commonpath([docs_root.resolve(), *navigation]))
    extra_anchors: dict[str, set[str]] = {}
    version_prefixes = []
    for index, (version, nav_file) in enumerate(zip(versions, navigation)):
        pages: dict[Path, str] = {}
        slug = (
            ''
            if index == 0
            else '/' + (version.get('slug') or _slug_from_title(version['display-name']))
        )
        version_prefixes.append(prefix + slug)
        _walk_nav(_load_nav(nav_file), prefix + slug, nav_file.parent, pages)
        if index:
            extra_anchors.update(
                (route, _collect_anchors(_strip_code(path.read_text())))
                for path, route in pages.items()
                if path.is_file()
            )
    findings = []
    for nav_file, version_prefix in zip(navigation, version_prefixes):
        source_root = nav_file.parent.parent if nav_file.parent.name == 'fern' else nav_file.parent
        findings.extend(
            _check_navigation(
                source_root=source_root,
                nav_file=nav_file,
                nav_root=nav_file.parent,
                prefix=version_prefix,
                report_root=report_root,
                redirects_file=docs_yml if nav_file == navigation[0] else None,
                published=published,
                extra_anchors=extra_anchors,
                repository_root=source_root.parent,
            )
        )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Validate local Markdown targets or Fern user-guide links.',
    )
    parser.add_argument(
        '--docs-root',
        type=Path,
        default=SCRIPT_DOCS_ROOT,
        help='Documentation root holding the MDX pages (default: the one containing this script).',
    )
    parser.add_argument(
        '--fern-dir',
        type=Path,
        default=None,
        help='Fern project directory (default: <docs-root>/fern).',
    )
    parser.add_argument(
        '--local-files-only',
        action='store_true',
        help='Check local targets in every authored Markdown file under the repository root.',
    )
    parser.add_argument('--published', action='store_true', help='reject unresolved source links')
    args = parser.parse_args()

    docs_root = args.docs_root.expanduser().resolve()
    if not docs_root.is_dir():
        print(f'error: documentation root not found: {docs_root}', file=sys.stderr)
        return 2

    try:
        if args.local_files_only:
            findings = check_local_file_links(docs_root.parent)
        else:
            fern_dir = (args.fern_dir or docs_root / 'fern').expanduser().resolve()
            findings = check(
                docs_root,
                fern_dir,
                published=args.published,
            )
    except ValueError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2

    for finding in findings:
        print(f'error: {finding}', file=sys.stderr)
    if findings:
        print(
            f'\nerror: found {len(findings)} broken documentation link(s).',
            file=sys.stderr,
        )
        return 1
    print('No broken documentation links found.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from docs.scripts.check_doc_links import check
from docs.scripts.check_doc_links import check_local_file_links
from docs.scripts.check_doc_links import load_page_routes
from docs.scripts.check_doc_links import prose_mask


class IsaacDocLinksTestCase(TestCase):
    """Verify Isaac site routes and repository-local link checks."""

    def test_yaml_navigation_preserves_quoted_titles_and_boolean_settings(self):
        with TemporaryDirectory() as directory:
            docs = Path(directory).resolve()
            fern = docs / 'fern'
            fern.mkdir()
            (fern / 'docs.yml').write_text('instances: [{url: example.com/isaac-os}]\n')
            (fern / 'index.yml').write_text(
                'navigation:\n  - section: Samples\n    skip-slug: true\n'
                '    contents: [{page: "Example #1", path: ../example.mdx}]\n'
            )
            page = docs / 'example.mdx'
            page.write_text('# Example\n[Self](/isaac-os/example-1#example)\n')
            self.assertEqual(load_page_routes(fern), {page: '/isaac-os/example-1'})
            self.assertEqual(check(docs, fern), [])

    def test_neutral_source_links_and_assembled_versions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            docs = root / 'docs'
            fern = docs / 'fern'
            fern.mkdir(parents=True)
            old = root / 'versions/v1/docs/fern'
            old.mkdir(parents=True)
            (root / 'source.h').write_text('// Source\n')
            (fern / 'docs.yml').write_text(
                'instances:\n  - url: example.com\nversions:\n'
                '  - display-name: Latest\n    path: ./index.yml\n'
                '  - display-name: v1\n    slug: v1\n'
                '    path: ../../versions/v1/docs/fern/index.yml\n'
            )
            (fern / 'index.yml').write_text('navigation:\n  - page: New\n    path: ../new.mdx\n')
            (docs / 'new.mdx').write_text('# New\n[Source](../source.h)\n')
            (old / 'index.yml').write_text('navigation:\n  - page: Old\n    path: ../old.mdx\n')
            page = old.parent / 'old.mdx'
            page.write_text('# Old\n[Explicit version](/v1/old#old)\n')
            with redirect_stdout(StringIO()):
                self.assertEqual(check(docs, fern), [])
                findings = check(docs, fern, published=True)
            self.assertTrue(any('source link' in finding.message for finding in findings))
            page.write_text('# Old\n[Unscoped own page](/old#old)\n')
            with redirect_stdout(StringIO()):
                findings = check(docs, fern)
            self.assertEqual(len(findings), 1)
            self.assertIn('no such page', findings[0].message)
            page.write_text('# Old\n[Missing in this version](/new#new)\n')
            with redirect_stdout(StringIO()):
                findings = check(docs, fern)
            self.assertEqual(len(findings), 1)
            self.assertIn('no such page', findings[0].message)
            page.write_text('# Old\n[Bad anchor](/v1/old#missing)\n')
            with redirect_stdout(StringIO()):
                findings = check(docs, fern)
            self.assertEqual(len(findings), 1)
            self.assertIn("no anchor 'missing'", findings[0].message)

    def test_isaac_routes_and_anchors(self) -> None:
        with TemporaryDirectory() as directory:
            docs = Path(directory) / 'docs'
            fern = docs / 'fern'
            fern.mkdir(parents=True)
            (fern / 'docs.yml').write_text(
                'instances:\n  - url: example.docs.buildwithfern.com/isaac-os\n',
                encoding='utf-8',
            )
            (fern / 'index.yml').write_text(
                'navigation:\n  - page: Guide\n    path: ../guide.mdx\n    slug: guide\n',
                encoding='utf-8',
            )
            page = docs / 'guide.mdx'
            page.write_text(
                '# Guide\n[valid](/isaac-os/guide#guide)\n',
                encoding='utf-8',
            )
            with redirect_stdout(StringIO()):
                self.assertEqual(check(docs, fern), [])

            page.write_text(
                '# Guide\n[broken](/isaac-os/guide#missing)\n',
                encoding='utf-8',
            )
            with redirect_stdout(StringIO()):
                findings = check(docs, fern)
            self.assertEqual(len(findings), 1)
            self.assertIn("no anchor 'missing'", findings[0].message)

            page.write_text(
                "# Guide\n<Card href='/isaac-os/guide#missing' />\n"
                "`<Card href='/isaac-os/guide#example' />`\n"
            )
            with redirect_stdout(StringIO()):
                findings = check(docs, fern)
            self.assertEqual(len(findings), 1)
            self.assertIn("no anchor 'missing'", findings[0].message)

    def test_repository_links_skip_site_routes_and_find_old_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / 'docs'
            docs.mkdir()
            (docs / 'guide.mdx').write_text('# Guide\n[site](/isaac-os/guide)\n', encoding='utf-8')
            readme = root / 'README.md'
            readme.write_text(
                '[guide](docs/guide.mdx)\n',
                encoding='utf-8',
            )
            with redirect_stdout(StringIO()):
                self.assertEqual(check_local_file_links(root), [])

            readme.write_text('[old guide](docs/guide.rst)\n', encoding='utf-8')
            with redirect_stdout(StringIO()):
                findings = check_local_file_links(root)
            self.assertEqual(len(findings), 1)
            self.assertIn('missing local target', findings[0].message)

    def test_repository_markdown_rejects_guide_routes_outside_code(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / '.copybara/checkout'
            snapshot.mkdir(parents=True)
            (snapshot / 'README.md').write_text('[Old](/isaac-os/example)\n')
            readme = root / 'README.md'
            for link in (
                '[Guide](/isaac-os/how-to/build-and-run#requirements)',
                '[guide]: </isaac-os/how-to/build-and-run?view=full>',
                "<a href='/isaac-os/how-to/build-and-run'>Guide</a>",
            ):
                with self.subTest(link=link):
                    readme.write_text(
                        '`[Inline](/isaac-os/example)`\n'
                        '```md\n[Fenced](/isaac-os/example)\n```\n'
                        '[External](https://example.com/isaac-os/guide)\n' + link + '\n'
                    )
                    with redirect_stdout(StringIO()):
                        findings = check_local_file_links(root)
                    self.assertEqual(len(findings), 1)
                    self.assertIn('use a relative source path', findings[0].message)

    def test_navigation_reports_missing_pages_and_duplicate_routes(self) -> None:
        with TemporaryDirectory() as directory:
            docs = Path(directory) / 'docs'
            fern = docs / 'fern'
            fern.mkdir(parents=True)
            (fern / 'docs.yml').write_text(
                'instances:\n  - url: example.docs.buildwithfern.com/isaac-os\n',
                encoding='utf-8',
            )
            (fern / 'index.yml').write_text(
                'navigation:\n'
                '  - page: First\n    path: ../first.mdx\n    slug: guide\n'
                '  - page: Second\n    path: ../missing.mdx\n    slug: guide\n',
                encoding='utf-8',
            )
            (docs / 'first.mdx').write_text('# First\n', encoding='utf-8')

            with redirect_stdout(StringIO()):
                findings = check(docs, fern)
            self.assertEqual(len(findings), 2)
            self.assertTrue(any('missing navigation page' in item.message for item in findings))
            self.assertTrue(any('duplicate navigation route' in item.message for item in findings))

    def test_link_diagnostics_and_redirects(self):
        with TemporaryDirectory() as directory:
            docs = Path(directory).resolve() / 'docs'
            fern = docs / 'fern'
            fern.mkdir(parents=True)
            (fern / 'index.yml').write_text('navigation: [{page: Guide, path: ../guide.mdx}]\n')
            (fern / 'docs.yml').write_text(
                'instances: [{url: example.com/isaac-os}]\n'
                'redirects:\n'
                '  - {source: /old, destination: /isaac-os/guide}\n'
                '  - {source: /external, destination: "https://example.com"}\n'
                '  - {source: /wild, destination: "/isaac-os/:path*"}\n'
                '  - {source: /broken, destination: /missing}\n'
            )
            page = docs / 'guide.mdx'
            cases = {
                '#guide': None,
                '?query': None,
                '/old': None,
                'https://example.com': None,
                '#missing': "no anchor 'missing' on this page",
                'missing.png': 'missing asset: missing.png',
                'guide.mdx#guide': 'page-file link: use the site path /isaac-os/guide#guide',
                'missing.mdx': 'link to a file outside the navigation: missing.mdx',
                'guide#guide': 'relative link is published unchanged: guide#guide '
                '(use /isaac-os/guide#guide)',
                'unknown': 'relative link is published unchanged: unknown '
                '(no page at /isaac-os/unknown; check the slugs in index.yml)',
                '/missing': 'no such page: /missing',
            }
            for target, message in cases.items():
                with self.subTest(target=target):
                    page.write_text(f'# Guide\n[Link]({target})\n')
                    with redirect_stdout(StringIO()):
                        findings = check(docs, fern)
                    expected = [] if message is None else [message]
                    expected.append('redirect /broken -> unknown page /missing')
                    self.assertEqual([item.message for item in findings], expected)

    def test_inline_code_mask_preserves_offsets_and_paragraph_boundaries(self):
        for source, expected in (
            ('a `code` b', 'a        b'),
            ('a ``one ` two`` b', 'a               b'),
            ('a `one\ntwo` b', 'a     \n     b'),
            ('a `one\n \ntwo` b', 'a `one\n \ntwo` b'),
            (r'a \`one` b', r'a \`one` b'),
            (r'a `one\` b', 'a        b'),
            ('a `` unmatched `code` b', 'a `` unmatched        b'),
        ):
            with self.subTest(source=source):
                self.assertEqual(prose_mask(source), expected)

    def test_unmatched_backtick_runs_remain_visible(self):
        source = ' '.join('x' + '`' * length for length in range(1, 2000))
        self.assertEqual(prose_mask(source), source)

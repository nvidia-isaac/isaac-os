#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import redirect_stderr
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from docs.scripts.check_doc_links import check
from docs.scripts.sync_application_pages import SourceContext
from docs.scripts.sync_application_pages import discover_pages
from docs.scripts.sync_application_pages import render
from docs.scripts.sync_application_pages import sync_pages


class ApplicationPageRoutesTestCase(TestCase):
    """Generated README links must follow changes to Fern configuration."""

    def test_generated_titled_links_are_validated(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            docs = root / 'docs'
            fern = docs / 'fern'
            fern.mkdir(parents=True)
            (fern / 'docs.yml').write_text(
                'instances:\n  - url: example.com/isaac-os\n', encoding='utf-8'
            )
            (fern / 'index.yml').write_text(
                'navigation:\n  - page: Guide\n    path: ../guide.mdx\n'
                '  - page: App\n    path: ../app.mdx\n',
                encoding='utf-8',
            )
            (docs / 'guide.mdx').write_text('# Guide\n', encoding='utf-8')
            source = root / 'README.md'
            for title in ('"Guide"', "'Guide'", '(Guide)'):
                for target, problem in (
                    ('docs/guide.mdx?view=full#guide', None),
                    ('docs/guide.mdx#missing', "no anchor 'missing'"),
                    ('/isaac-os/missing', 'no such page'),
                ):
                    with self.subTest(title=title, target=target):
                        source.write_text(f'[guide]({target} {title})\n', encoding='utf-8')
                        (docs / 'app.mdx').write_text(render(source, root), encoding='utf-8')
                        with redirect_stdout(StringIO()):
                            findings = check(docs, fern)
                        if problem is None:
                            self.assertEqual(findings, [])
                        else:
                            self.assertEqual(len(findings), 1)
                            self.assertIn(problem, findings[0].message)

    def test_navigation_and_instance_changes_update_links(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fern = root / 'docs' / 'fern'
            fern.mkdir(parents=True)
            guide = root / 'docs' / 'guide.mdx'
            guide.write_text('# Usage\n', encoding='utf-8')
            source = root / 'README.md'
            source.write_text('[guide](docs/guide.mdx#usage)\n', encoding='utf-8')
            for prefix, section, slug in (
                ('isaac-os', 'guides', 'original'),
                ('renamed-site', 'tutorials', 'renamed-page'),
            ):
                with self.subTest(prefix=prefix, section=section, slug=slug):
                    (fern / 'docs.yml').write_text(
                        f'instances:\n  - url: example.com/{prefix}\n', encoding='utf-8'
                    )
                    (fern / 'index.yml').write_text(
                        f'navigation:\n  - section: Guides\n    slug: {section}\n'
                        '    contents:\n      - page: Guide\n        path: ../guide.mdx\n'
                        f'        slug: {slug}\n',
                        encoding='utf-8',
                    )
                    self.assertIn(
                        f'[guide](/{prefix}/{section}/{slug}#usage)', render(source, root)
                    )

    def test_link_titles_queries_and_fragments(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fern = root / 'docs' / 'fern'
            fern.mkdir(parents=True)
            (fern / 'docs.yml').write_text(
                'instances:\n  - url: example.com/isaac-os\n', encoding='utf-8'
            )
            (fern / 'index.yml').write_text(
                'navigation:\n  - page: Guide\n    path: ../guide.mdx\n    slug: guide\n',
                encoding='utf-8',
            )
            (root / 'docs' / 'guide.mdx').write_text('# Usage\n', encoding='utf-8')
            source = root / 'README.md'
            for title in ('', ' "Usage guide"', " 'Usage guide'", ' (Usage guide)'):
                for target in (
                    'docs/guide.mdx?view=full#usage',
                    '<docs/guide.mdx?view=full#usage>',
                ):
                    with self.subTest(title=title, target=target):
                        source.write_text(f'[guide]({target}{title})\n', encoding='utf-8')
                        expected = target.replace('docs/guide.mdx', '/isaac-os/guide')
                        self.assertIn(f'[guide]({expected}{title})', render(source, root))


class PublicationLinksTestCase(TestCase):
    """Guide and source links must work in both generated and neutral pages."""

    def setUp(self):
        self.directory = TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        fern = self.root / 'docs/fern'
        fern.mkdir(parents=True)
        (fern / 'docs.yml').write_text('instances:\n  - url: example.com/isaac-os\n')
        (fern / 'index.yml').write_text(
            'navigation:\n  - page: Guide\n    path: ../guide.mdx\n    slug: guide\n'
        )
        (self.root / 'docs/guide.mdx').write_text('# Usage\n')
        operator = self.root / 'operators/demo'
        operator.mkdir(parents=True)
        (operator / 'README.md').write_text('# Contract\n')
        (operator / 'demo file.h').write_text('// Header\n')
        self.source = self.root / 'README.md'

    def test_publication_maps_guide_and_source_targets(self):
        self.source.write_text(
            '[Root](.)\n'
            '[Guide](docs/guide.mdx?view=full#usage "Guide")\n'
            '[Header](operators/demo/demo%20file.h?plain=1#L1 "Header")\n'
            '[Contract](operators/demo/README.md#contract)\n'
            '[Directory](operators/demo)\n'
            '[Reference][header]\n[header]: operators/demo/demo%20file.h#L1\n'
        )
        for repository, ref, segment in (
            ('https://github.com/nvidia-isaac/isaac-os', 'v1.2.3', ''),
            ('https://gitlab.example/group/isaac-os', 'docs/links', '/-'),
        ):
            with self.subTest(repository=repository):
                output = render(self.source, self.root, context=SourceContext(repository, ref))
                self.assertIn('[Guide](/isaac-os/guide?view=full#usage "Guide")', output)
                base = f'{repository}{segment}'
                self.assertIn(f'[Root]({base}/tree/{ref}/)', output)
                self.assertIn(
                    f'[Header]({base}/blob/{ref}/operators/demo/demo%20file.h?plain=1#L1 "Header")',
                    output,
                )
                self.assertIn(
                    f'[Contract]({base}/blob/{ref}/operators/demo/README.md#contract)', output
                )
                self.assertIn(f'[Directory]({base}/tree/{ref}/operators/demo)', output)
                self.assertIn(
                    f'[header]: {base}/blob/{ref}/operators/demo/demo%20file.h#L1', output
                )

    def test_rendering_leaves_code_examples_unchanged(self):
        self.source.write_text(
            '`[Inline](missing.md)`\n'
            '``[Backticks](missing.md)``\n'
            '```md\n[Fenced](missing.md)\n![Image](missing.png)\n```\n'
            '~~~~md\n[Reference]: missing.md\n~~~~\n'
        )
        self.assertTrue(render(self.source, self.root).endswith(self.source.read_text()))

    def test_unmatched_or_escaped_backtick_does_not_hide_prose_links(self):
        for opening in ('A stray ` character.', r'An escaped \` character.'):
            with self.subTest(opening=opening):
                self.source.write_text(opening + '\n\n[Missing](missing.md)\n\n`code`\n')
                with self.assertRaisesRegex(ValueError, 'missing or out-of-tree'):
                    render(self.source, self.root)

    def test_neutral_reference_and_html_links_still_resolve_from_generated_page(self):
        self.source.write_text(
            '# App\n[Header][source]\n[source]: operators/demo/demo%20file.h#L1\n'
            '<a href="operators/demo/README.md#contract">Contract</a>\n'
        )
        page = self.root / 'docs/app.mdx'
        with (self.root / 'docs/fern/index.yml').open('a') as nav:
            nav.write('  - page: App\n    path: ../app.mdx\n')
        page.write_text(render(self.source, self.root, {self.source: page}))
        with redirect_stdout(StringIO()):
            findings = check(self.root / 'docs', self.root / 'docs/fern')
        self.assertEqual([str(finding) for finding in findings], [])

    def test_publication_rejects_missing_or_escaping_targets(self):
        context = SourceContext('https://github.com/nvidia-isaac/isaac-os', 'v1.2.3')
        for target in ('missing.md', '../outside.md'):
            with self.subTest(target=target):
                self.source.write_text(f'[Missing]({target})\n')
                with self.assertRaisesRegex(ValueError, 'missing or out-of-tree'):
                    render(self.source, self.root, context=context)


class ApplicationDiscoveryTestCase(TestCase):
    """Discover nested application READMEs without registering individual paths."""

    def test_discovery_cross_links_images_and_drift(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fern = root / 'docs' / 'fern'
            fern.mkdir(parents=True)
            (fern / 'docs.yml').write_text(
                'instances:\n  - url: example.com/isaac-os\n', encoding='utf-8'
            )
            nav = fern / 'index.yml'
            nav.write_text(
                'navigation:\n  - page: App\n    path: ../applications/demo.mdx\n'
                '  - page: Simulation\n    path: ../applications/demo/sim.mdx\n',
                encoding='utf-8',
            )
            app = root / 'applications/demo/README.md'
            simulation = root / 'applications/demo/sim/README.md'
            simulation.parent.mkdir(parents=True)
            image = app.parent / 'example.png'
            image.write_bytes(b'image fixture')
            app.write_text(
                '[simulation](sim/README.md#usage)\n![Example](example.png)\n'
                '[source directory](sim)\n',
                encoding='utf-8',
            )
            simulation.write_text('# Usage\n', encoding='utf-8')
            operator = root / 'operators/demo/README.md'
            operator.parent.mkdir(parents=True)
            operator.write_text('# Not an application\n', encoding='utf-8')
            with redirect_stdout(StringIO()):
                self.assertEqual(sync_pages(root, check_only=True), 1)
                self.assertFalse((root / 'docs/applications').exists())
                self.assertEqual(sync_pages(root), 0)
                self.assertEqual(sync_pages(root, check_only=True), 0)
                generated = (root / 'docs/applications/demo.mdx').read_text(encoding='utf-8')
                self.assertIn('[simulation](/isaac-os/simulation#usage)', generated)
                self.assertIn('![Example](../../applications/demo/example.png)', generated)
                self.assertIn('source directory (`applications/demo/sim`)', generated)
                self.assertFalse((root / 'docs/operators').exists())
                simulation.write_text('# Usage\nUpdated content.\n', encoding='utf-8')
                self.assertEqual(sync_pages(root, check_only=True), 1)
                self.assertEqual(sync_pages(root), 0)
                self.assertEqual(sync_pages(root, check_only=True), 0)

            # A new app is discovered automatically; it must be in navigation
            # before generation can succeed. No partial writes on failure.
            added = root / 'applications/new/README.md'
            added.parent.mkdir()
            added.write_text('# New app\n', encoding='utf-8')
            with redirect_stderr(StringIO()) as errors:
                self.assertEqual(sync_pages(root), 1)
                self.assertIn('missing from Fern navigation', errors.getvalue())
            self.assertFalse((root / 'docs/applications/new.mdx').exists())
            with nav.open('a', encoding='utf-8') as stream:
                stream.write('  - page: New\n    path: ../applications/new.mdx\n')
            with redirect_stdout(StringIO()):
                self.assertEqual(sync_pages(root), 0)
                self.assertEqual(sync_pages(root, check_only=True), 0)


class ApplicationLifecycleTestCase(TestCase):
    """Cover removed sources and image syntax when relocating application pages."""

    def setUp(self) -> None:
        """Create a minimal application and its documentation navigation."""
        # unittest cleanup keeps the fixture alive through each test.
        self.directory = TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        fern = self.root / 'docs/fern'
        fern.mkdir(parents=True)
        (fern / 'docs.yml').write_text('instances:\n  - url: example.com/isaac-os\n')
        (fern / 'index.yml').write_text(
            'navigation:\n  - page: Demo\n    path: ../applications/demo.mdx\n'
        )
        self.source = self.root / 'applications/demo/README.md'
        self.source.parent.mkdir(parents=True)
        self.source.write_text('# Demo\n')
        self.page = self.root / 'docs/applications/demo.mdx'

    def test_campaign_scripts_subtree_is_not_published(self) -> None:
        """Exclude all campaign docs while retaining application and other nested pages."""
        benchmark = self.root / 'applications/robot_control_benchmark/README.md'
        internal = benchmark.parent / 'scripts/README.md'
        nested = self.source.parent / 'scripts/README.md'
        for source in (benchmark, internal, nested):
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text('# Documentation\n')
        for directory in (internal.parent, internal.parent / 'guides/nested'):
            directory.mkdir(parents=True, exist_ok=True)
            for name in ('README.md', 'BENCHMARK.md', 'RESULTS.md'):
                (directory / name).write_text('# Internal documentation\n')
        with (self.root / 'docs/fern/index.yml').open('a') as stream:
            stream.write(
                '  - page: Benchmark\n    path: ../applications/robot_control_benchmark.mdx\n'
                '  - page: Demo scripts\n    path: ../applications/demo/scripts.mdx\n'
            )

        self.assertEqual(set(discover_pages(self.root)), {self.source, benchmark, nested})
        with redirect_stdout(StringIO()):
            self.assertEqual(sync_pages(self.root), 0)
            self.assertEqual(sync_pages(self.root, check_only=True), 0)
        self.assertEqual(
            set((self.root / 'docs/applications').rglob('*.mdx')),
            {
                self.page,
                self.root / 'docs/applications/robot_control_benchmark.mdx',
                self.root / 'docs/applications/demo/scripts.mdx',
            },
        )

    def test_benchmark_and_results_sources(self) -> None:
        """Render optional app records, rewrite cross-links and detect drift/collisions."""
        benchmark = self.source.with_name('BENCHMARK.md')
        results = self.source.with_name('RESULTS.md')
        self.source.write_text('# Demo\n[Benchmark](BENCHMARK.md)\n')
        benchmark.write_text('# Benchmark\n[Results](RESULTS.md#test-details)\n')
        results.write_text('# Results\n\n## Test details\n')
        nav = self.root / 'docs/fern/index.yml'
        with nav.open('a') as stream:
            stream.write(
                '  - page: Benchmark\n    path: ../applications/demo/benchmark.mdx\n'
                '  - page: Results\n    path: ../applications/demo/results.mdx\n'
            )
        with redirect_stdout(StringIO()):
            self.assertEqual(sync_pages(self.root), 0)
            self.assertIn('[Benchmark](/isaac-os/benchmark)', self.page.read_text())
            self.assertIn(
                '[Results](/isaac-os/results#test-details)',
                (self.root / 'docs/applications/demo/benchmark.mdx').read_text(),
            )
            self.assertEqual(sync_pages(self.root, check_only=True), 0)
            results.write_text('# Results\n\n## Test details\nUpdated values.\n')
            self.assertEqual(sync_pages(self.root, check_only=True), 1)
            self.assertEqual(sync_pages(self.root), 0)
            self.assertEqual(sync_pages(self.root, check_only=True), 0)
        # A nested README must not silently overwrite the optional record page.
        duplicate = self.source.parent / 'benchmark/README.md'
        duplicate.parent.mkdir()
        duplicate.write_text('# Different benchmark\n')
        with redirect_stderr(StringIO()) as errors:
            self.assertEqual(sync_pages(self.root), 1)
        self.assertIn('multiple sources', errors.getvalue())

    def test_deleted_and_renamed_readmes_leave_reported_orphans(self) -> None:
        """Fail before writing, preserving orphan files for explicit cleanup."""
        with redirect_stdout(StringIO()):
            self.assertEqual(sync_pages(self.root), 0)
        original = self.page.read_bytes()
        self.source.unlink()
        for renamed in (False, True):
            if renamed:
                new = self.root / 'applications/renamed/README.md'
                new.parent.mkdir()
                new.write_text('# Renamed\n')
            for check_only in (False, True):
                with self.subTest(renamed=renamed, check_only=check_only):
                    with redirect_stderr(StringIO()) as errors:
                        self.assertEqual(sync_pages(self.root, check_only=check_only), 1)
                    self.assertIn('docs/applications/demo.mdx', errors.getvalue())
                    self.assertIn('no source Markdown', errors.getvalue())
                    self.assertEqual(self.page.read_bytes(), original)
        # Authored pages are not mistaken for generated orphans.
        self.page.write_text('# Authored page\n')
        new.unlink()
        with redirect_stdout(StringIO()):
            self.assertEqual(sync_pages(self.root, check_only=True), 0)

    def test_image_destinations_and_titles(self) -> None:
        """Preserve titles and URL suffixes while relocating varied image paths."""
        for name in ('image.png', 'image (one).png', 'image(one(two)).png', 'image).png'):
            (self.source.parent / name).write_bytes(b'image fixture')
        for target, expected in (
            ('image.png', 'image.png'),
            ('<image (one).png>', 'image%20%28one%29.png'),
            ('image(one(two)).png', 'image%28one%28two%29%29.png'),
            (r'image\).png', 'image%29.png'),
            ('<image).png>', 'image%29.png'),
            ('image.png?view=full#detail', 'image.png?view=full#detail'),
        ):
            for title in ('', ' "Example"', " 'Example'", ' (Example)'):
                with self.subTest(target=target, title=title):
                    self.source.write_text(f'![Example\nimage](  {target}{title}  )\n')
                    result = render(self.source, self.root, {self.source: self.page})
                    self.assertIn(
                        f'![Example\nimage](../../applications/demo/{expected}{title})', result
                    )
                    self.page.parent.mkdir(parents=True, exist_ok=True)
                    self.page.write_text(result)
                    with redirect_stdout(StringIO()):
                        self.assertEqual(check(self.root / 'docs', self.root / 'docs/fern'), [])
        for target in (
            'https://example.com/image.png',
            '/assets/image.png',
            'data:image/png;base64,x',
        ):
            markup = f'![Example]( {target} "Title" )'
            self.source.write_text(markup)
            self.assertIn(markup, render(self.source, self.root, {self.source: self.page}))

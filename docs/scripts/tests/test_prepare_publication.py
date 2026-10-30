# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publication must use original Markdown and the selected content tree."""

from pathlib import Path
import tempfile
import unittest

from docs.scripts import prepare_publication as publication
from docs.scripts import sync_application_pages as pages
from docs.scripts.prepare_publication import prepare_copy


class PreparationTest(unittest.TestCase):
    def test_prompt_links_are_absolute_without_changing_examples_or_other_links(self):
        content = """[Ordinary](/guide)
```mdx
<Prompt>[Example](/example)</Prompt>
```
`<Prompt>[Inline](/inline)</Prompt>`
<Prompt title="Empty" />
[Between](/ordinary)
<Prompt title="✨ Agent prompt: Test" hidePrompt actions={["cursor"]}>
[For agents](/isaac-os/for-agents)
[Guide](</version/guide?view=full#usage> "Guide")
[Markdown](/version/manual.md?view=full#section)
[PDF](/isaac-os/files/spec.pdf)
[Skill](https://github.com/example/repo/blob/v1/skills/test/SKILL.md)
[External](//example.com/other)
![Image](/image.png)
`[Code](/code)`
</Prompt>
<llms-ignore>
<Prompt>[Second](/version/other)</Prompt>
</llms-ignore>
"""
        result = publication.resolve_prompt_links(content, 'https://docs.example.com/base')
        expected = (
            content.replace(
                '</version/guide?view=full#usage>',
                '<https://docs.example.com/version/guide.md?view=full#usage>',
            )
            .replace(
                '](/version/manual.md?view=full#section)',
                '](https://docs.example.com/version/manual.md?view=full#section)',
            )
            .replace(
                '](/isaac-os/files/spec.pdf)',
                '](https://docs.example.com/isaac-os/files/spec.pdf)',
            )
            .replace(
                '](/isaac-os/for-agents)',
                '](https://docs.example.com/isaac-os/for-agents.md)',
            )
            .replace('](/version/other)', '](https://docs.example.com/version/other.md)')
        )
        self.assertEqual(result, expected)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'source'
        fern = self.root / 'docs/fern'
        fern.mkdir(parents=True)
        (fern / 'docs.yml').write_text('instances:\n  - url: example.com/isaac-os\n')
        (fern / 'index.yml').write_text(
            'navigation:\n  - page: Demo\n    path: ../applications/demo.mdx\n'
            '  - page: Guide\n    path: ../guide.mdx\n'
        )
        app = self.root / 'applications/demo'
        app.mkdir(parents=True)
        (app / 'README.md').write_text('# Demo\n[Source](demo.h#L1)\n')
        (app / 'demo.h').write_text('// Header\n')
        (self.root / 'docs/guide.mdx').write_text(
            '# Guide\n[Source](../applications/demo/demo.h#L1)\n'
            '[Demo](/isaac-os/demo#demo)\n'
            '`[Example](not-a-file.h)`\n'
            '<llms-ignore>\n'
            '<Prompt title="✨ Agent prompt: Guide" hidePrompt actions={["cursor"]}>\n'
            '[Guide](/isaac-os/guide)\n'
            '</Prompt>\n'
            '</llms-ignore>\n'
            '<release_tag>\n'
        )
        self.assertEqual(pages.sync_pages(self.root), 0)

    def test_preparation_regenerates_markdown_and_resolves_authored_links(self):
        output = self.root.parent / 'publication'
        context = pages.SourceContext('https://github.com/nvidia-isaac/isaac-os', 'v1.2.3')
        prepare_copy(self.root, output, context)
        link = 'https://github.com/nvidia-isaac/isaac-os/blob/v1.2.3/applications/demo/demo.h#L1'
        self.assertIn(link, (output / 'docs/applications/demo.mdx').read_text())
        guide = (output / 'docs/guide.mdx').read_text()
        self.assertIn(link, guide)
        self.assertIn('[Demo](/isaac-os/demo#demo)', guide)
        self.assertIn('`[Example](not-a-file.h)`', guide)
        self.assertIn('[Guide](/isaac-os/guide)', guide)
        self.assertNotIn('<release_tag>', guide)
        self.assertNotIn('github.com', (self.root / 'docs/applications/demo.mdx').read_text())

    def test_preview_keeps_release_tag_placeholder(self):
        output = self.root.parent / 'publication'
        context = pages.SourceContext('https://gitlab.example.com/isaac-os', 'feature')
        prepare_copy(self.root, output, context)
        self.assertIn('<release_tag>', (output / 'docs/guide.mdx').read_text())

    def test_missing_public_file_is_not_filled_from_other_checkout(self):
        (self.root / 'applications/demo/demo.h').unlink()
        with self.assertRaisesRegex(ValueError, 'missing or out-of-tree'):
            prepare_copy(
                self.root,
                self.root.parent / 'publication',
                pages.SourceContext('https://github.com/nvidia-isaac/isaac-os', 'v1.2.3'),
            )

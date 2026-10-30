# Isaac OS documentation (Fern)

User-guide pages are committed as **MDX** (`.mdx`) under `docs/`. Fern configuration
lives under `docs/fern/` (`fern.config.json`, `docs.yml`, `index.yml`, `assets/`,
`dist/`).

C++ API pages are **not** included in this first Fern iteration. The Doxygen
sources under `docs/api/` and `docs/Doxyfile` remain for a later pass.

## Scripts

Start with `docs/build.sh` to check or preview documentation locally.

| Script | Role |
| --- | --- |
| [build.sh](build.sh) | Run source checks and Fern validation or a local preview. |
| [sync_application_pages.py](scripts/sync_application_pages.py) | Generate application MDX from Markdown, or check for drift. |
| [check_doc_links.py](scripts/check_doc_links.py) | Validate guide routes, anchors, assets and repository links. |
| [prepare_publication.py](scripts/prepare_publication.py) | Prepare a temporary docs tree with source links for a selected repository and ref. |



## Diataxis authoring rules

Follow [Diataxis](https://diataxis.fr/). Choose each page's primary reader need before editing:

| Type | Purpose | Write |
| --- | --- | --- |
| Tutorial | Learn by completing a guided experience | A bounded outcome, prerequisites, steps and observable success |
| How-to | Accomplish a specific task | Direct instructions for a reader who knows the basics |
| Explanation | Understand why | Concepts, reasoning, relationships and tradeoffs |
| Reference | Look up what exists | Accurate descriptions of interfaces, configuration, behavior and limits |

- Classify by purpose, not by the presence of code or an architectural diagram.
  Concrete application wiring and class descriptions usually belong in Reference;
  design reasoning belongs in Explanation.
- Match the body, title, navigation and authored source directory to that purpose.
  Renaming or moving a page alone does not change its type. Explanation and Reference
  must not become follow-along procedures with numbered "Step" headings.
- Link to other page types for substantial material serving another need. Keep only
  the context needed here; tutorials may repeat essential setup to remain self-contained.
- Application Reference pages retain exact build/run recipes beside their mode and
  configuration contracts. Keep shared setup in How-to and guided learning in Tutorials.
- Keep one canonical home for detailed contracts and procedures. Name references for
  the implementation they describe; do not imply generic coverage from one robot example.
- Assess product scope separately from Diataxis type. Fitting Reference or Explanation
  does not justify a standalone guide page or top-level entry. Embodiment-specific behavior
  must remain clearly scoped to that embodiment.
- Keep narrow details in application docs or component READMEs beside the source, and repository
  orientation in the root README. Prefer an existing relevant section or local accordion to a
  new page, extra navigation nesting, or a miscellaneous category.
- Create pages only for distinct current reader needs, not to fill all four categories. When
  retaining mixed legacy content temporarily, record the deferred cleanup explicitly.

Start most authored pages with a short `<Note>` stating the outcome or lookup scope
and the main steps or concepts. Keep limits that affect the reader’s choice here;
do not repeat the summary below it. Overview and generated Applications use their
own introductions. Incomplete pages combine the summary with their existing Warning.

Maintained How-to guides place one task-specific `<Prompt>` after the summary and
any required warning. Start the prompt with "Find &lt;task inputs&gt; in our conversation
and project. Ask me for missing details." Put compatibility checks before setup
changes, and keep their requirements in the linked guide. Follow the Prose rule in
[AGENTS.md](../AGENTS.md). Use `title="✨ Agent prompt: <task>"`, `hidePrompt`, and the native
`actions={["cursor"]}`. Isaac OS runs locally, so offer only the desktop Cursor action.
Link the guide's published route as a
Markdown link and, when relevant, its skill using a relative source path. Do not add
`.md`; publication adds it. Keep incomplete workflows out of executable prompts; no
task prompt is needed on the Yocto outline. Wrap each `<Prompt>` in `<llms-ignore>` so it
stays out of agent Markdown.

Overview has one prompt that points to For agents.

## Incomplete pages and benchmark records

Incomplete pages may stay visible in the navigation. Keep their sidebar labels
unmarked. Below the title, write only `Coming soon.` Incomplete sections of a page
use a `<Warning>` that says the content will come in a future release. Write for
readers: no internal labels such as priorities ("P1") or review notes. Do not mark
every maintained page merely because it could improve. Remove the notice when the
page or section has usable steps and observable success.

Keep the generic benchmarking method in How-to and worked learning experiences in
Tutorials. Combine each application's benchmark record and results in one **Benchmark**
page: scope first, then **Results**, then **Run the benchmark**. Put run details in a
**Jetson AGX Thor Benchmark Configuration** accordion under **Results**.
Keep **Run the benchmark** focused on runnable commands with brief app-specific explanations.
Link to [Profile an Application](how-to/profile_an_application.mdx) for shared profiling and
reporting details. Preserve exact paths, capture settings and measurement windows in the commands.
Application benchmark captures and published measurements target AGX Thor only.
Keep the test date, tested versions, settings and material limitations beside
the measurements. Identify the tested Isaac OS revision by its full commit SHA
(`git rev-parse HEAD`), not by merge request number or internal tag. When the
commit exists only in the internal repository, add `(internal mirror)` after it.
Label missing details explicitly; do not infer a test date or commit from when its
documentation was committed. A changed path or timing boundary requires reanalyzing
the captures or collecting new measurements; never relabel old numbers.

Write `<release_tag>` where readers need the Isaac OS release tag, for example in
`git clone --branch`. Publication replaces it with the tag of each
published docs version. MR previews keep the placeholder.

## Edit documentation

1. Change `*.mdx` files here. For generated application pages, edit
   `applications/**/README.md` or `BENCHMARK.md`, or `RESULTS.md` when the app keeps
   a separate benchmark history. Then run `python3 docs/scripts/sync_application_pages.py`.
   Pre-commit checks source and generated-page changes for drift.
2. Update sidebar entries in `fern/index.yml` when adding or renaming pages.
   Renaming a page changes its published URL unless you pin the old one with
   `slug:`.
3. Follow the [link rules](#write-links) below.
4. The root `README.md` mirrors the `docs/overview.mdx` sections: Introduction,
   Performance, Try it now and Go further. Change both files together. Keep the same
   text and numbers, and the same commands where both files show them; change only
   the format (`README.md` tables and code blocks for Fern cards and tabs) and the
   link style. In Try it now, `README.md` shows only the first tab and links to
   Getting started for the others. When published measurements change, update the
   Performance numbers in both files. The badges at the top of `README.md` are for
   repository visitors only; the site shows the agent prompt instead.

## Write links

| Target | README / Markdown | Authored MDX |
| --- | --- | --- |
| Our source file or directory | Relative path | Relative path |
| Our guide page | Internal docs preview URL in `README.md` (Copybara maps it to `docs.nvidia.com/isaac-os`); relative path to its source elsewhere | Published `/isaac-os/...` route |
| Section on the same page | `#anchor` | `#anchor` |
| Dependency overview | Public project or docs URL | Public project or docs URL |
| Dependency code or API | Public URL for the applicable dependency version | Public URL for the applicable dependency version |

`README.md` links point to the latest docs, not the checked-out version.

The `check-markdown-links` pre-commit hook checks local targets in Markdown and MDX.
It rejects `/isaac-os/...` routes in Markdown. Code examples are ignored.

Authored MDX uses [Fern's supported site paths](https://buildwithfern.com/learn/docs/writing-content/markdown-basics)
for guide pages so links also work in ordinary local previews, which skip publication preprocessing.

For example, link from `docs/how-to/` to `[Dockerfile](../../Dockerfile)`,
not a GitHub or GitLab URL containing `main` or `dev`. Publication selects the
GitLab branch for internal docs and the GitHub tag for public docs. Guide links
retain the selected docs version. Copybara does not translate these URLs.

Keep explicit tag or commit links when citing measured results or other evidence.
Internal commit IDs may not exist in the public repository after Copybara.
Clone commands and code examples keep their explicit repository URLs.
Until a dependency reference is public, keep its internal URL inside an
internal-only block; replace it with the public versioned URL when available.

## Add an application page

The generator discovers these files recursively under `applications/`:

| Source | Generated page |
| --- | --- |
| `<app>/README.md` | `docs/applications/<app>.mdx` |
| `<app>/BENCHMARK.md` | `docs/applications/<app>/benchmark.mdx` |

`BENCHMARK.md` is optional. Group it with the app as **Overview** and **Benchmark**.
The Benchmark page contains results plus the configurations, commands and analysis
needed to interpret and reproduce them. Guided benchmarking belongs in How-to or
Tutorials. Nested READMEs retain their directory paths; operator READMEs are excluded.

The generator also supports `RESULTS.md` as `<app>/results.mdx` for substantial
benchmark histories that warrant a separate page. Use the combined page by default.

1. Write the application's Markdown sources.
2. Add its generated page path to `fern/index.yml`, choosing its sidebar title
   and published slug. The generator reports pages missing from navigation.
3. Run `python3 docs/scripts/sync_application_pages.py`, then `docs/build.sh`.
   Commit the sources, generated pages and navigation changes together.

Links between application Markdown files become links to their published Fern pages.
Relative image paths are adjusted for the generated page's location. Other
repository links remain readable source paths in committed MDX. Publication
regenerates these links from the original README: published targets open their
guide pages; other files and directories open the source repository at that
version's branch or tag. The `check-application-docs-sync`
pre-commit hook matches all three application filenames at any depth, generated pages, Fern
configuration, and generator/link-checker changes; adding an application needs
no hook change. When removing an application, also remove its generated page
and navigation entry. Both generation and `--check` reject orphaned generated
pages until they are removed.

## Build and preview

Run the following commands from the Isaac OS repository root. You need Python 3
on the host and, by default, a running Docker daemon. The wrapper runs the Python
checks on the host before invoking Fern in Docker. Install their dependencies
with `python3 -m pip install -r docs/requirements.txt`. The docs image supplies the
pinned Fern CLI; no host Fern installation or Isaac OS application build is needed.
On macOS with Colima, keep the checkout in a directory shared with the VM.
The docs image includes `lsof`, which Fern uses to stop the preview on `Ctrl+C`.

### Validate documentation

Validate locally using the Docker image:

```bash
docs/build.sh
```

The wrapper checks that generated pages match their application Markdown
sources, runs `docs/scripts/check_doc_links.py` for Fern routes and repository
links, then runs `fern check --local --warnings`. The link checks catch targets
that Fern's local syntax check can miss. A successful run exits with status 0;
this command validates the sources without exporting an HTML site or publishing
a hosted preview.

If generated application pages are stale, regenerate them and rerun validation:

```bash
python3 docs/scripts/sync_application_pages.py
docs/build.sh
```

### Preview documentation

Preview with a local Fern docs server:

```bash
docs/build.sh preview
```

The wrapper runs the same Python checks before starting the server. Open the
local URL printed by Fern (default port: `3000`). Keep the command running while
previewing; stop it with `Ctrl+C`. To use another port (avoid `3001`, which Fern
uses for its backend):

```bash
docs/build.sh --port 3002 preview
```

To preview source-browser links, supply the repository and ref explicitly:

```bash
docs/build.sh preview \
  --source-repository https://github.com/nvidia-isaac/isaac-os \
  --source-ref v1.2.3
```

Use a ref that contains the files you are previewing. This prepares a temporary
copy with your local edits; committed MDX stays unchanged.
Restart this contextual preview after further source edits.

This serves your local checkout without publishing or updating a hosted MR preview.
An unauthenticated local preview omits the shared NVIDIA theme; you can still
check the pages and navigation. To load the theme, log in and restart the preview:

```bash
docs/build.sh login
docs/build.sh --port 5901 preview
```

Open the login link printed by Fern and approve the displayed code. The Docker
wrapper reuses credentials from `~/.fern` across runs. It also forwards
`FERN_TOKEN` when set in the host environment.

### Run without Docker

Install Node.js and npm on the host (the docs image uses Node.js 22), then install
the pinned CLI and use `--no-docker`. Python 3 and `lsof` must be on `PATH`;
macOS provides `lsof` in `/usr/sbin`:

```bash
npm install -g fern-api@5.168.0 pnpm@9.15.9
docs/build.sh --no-docker
docs/build.sh --no-docker preview
```

Keep the CLI version aligned with `docs/fern/fern.config.json` and `docs/Dockerfile`.
The pinned `pnpm` supplies Fern's local preview bundle dependencies. Installing it
up front avoids a global package installation when starting the preview server.
Run `docs/build.sh --help` for all wrapper options.

For direct Fern commands, run from **`docs`** so the CLI discovers the
configuration under `fern/`. These commands bypass the wrapper's Python checks:

```bash
cd docs
fern check --local --warnings
fern docs dev
```

## Adding content

Create Markdown/MDX pages under the appropriate directory:

- `docs/tutorials/` - step-by-step tutorials
- `docs/how-to/` - task-oriented how-to guides, including instrumentation and the
  generic benchmarking method
- `docs/explanation/` - explanations
- `docs/applications/` - generated reference application pages
- `docs/reference/` - authored reference pages for the CLI, components, platform support,
  and system configuration

Add new pages to the navigation in `fern/index.yml`.

Mermaid diagrams use fenced `mermaid` code blocks. Keep NVIDIA palette
`classDef` values in each diagram; Fern renders Mermaid natively.

Application READMEs render in both repository Markdown and Fern. Use
`> **Note:**` or `> **Warning:**` for callouts, and `<details><summary>` for optional
rationale or troubleshooting, with blank lines around the Markdown body. Keep
linked headings, required steps and safety limits outside `<details>`.

Prefix troubleshooting disclosure titles with ⚠️, sample log/output titles with 📄,
and optional explanations with ❓. Phrase explanation titles as questions. Use the
same labels for Fern `<Accordion>` and Markdown `<summary>`; leave procedural
alternatives unmarked. Keep brief success cues visible and collapse only the details.

Mode tabs may group complete run instructions, including that mode's warnings.
Keep shared requirements and mode comparisons outside. Give each `<Tab>` a unique
`id` and an explicit Markdown heading so GitHub/GitLab show readable stacked
sections; leave the body unindented. Check direct links and the table of contents
in the pinned Fern preview. Omit `language` to avoid site-wide tab synchronization.

---
paths:
  - "docs/for_agents.mdx"
  - "skills/**"
  - ".agents/skills/**"
  - "**/SKILL.md"
---

# Skill Authoring Rules


Apply these rules while writing a skill and validate it locally, so compliance findings surface
during development rather than at publication time.

The authoritative policy is the
[SkillEvaluator documentation](https://docs.nvidia.com/skills/skillevaluator). When it disagrees
with this rule, follow it and update this rule.



## Layout

Put each skill in the directory for its purpose, even if you started it in a different location:

- Skills for Isaac OS users (product skills): `skills/<name>/SKILL.md`.
- Skills for repository maintenance (contributor skills): `.agents/skills/<name>/SKILL.md`.

For every skill:

- Match the directory name to the frontmatter `name`.
- Keep `SKILL.md` under 500 lines. Move detail into `references/`, helpers into `scripts/`, and
  static files into `assets/`, linked directly from `SKILL.md` without nested reference chains.
- Every referenced file must exist in the repository. Use repository-relative paths, never
  contributor-specific home paths.
- Do not commit generated validation reports, skill cards, benchmarks, or signatures.


## Frontmatter

```yaml
---
name: example-skill
description: >
  What the skill does and when to use it, with the keywords an agent needs to select it.
license: Apache-2.0
metadata:
  author: "Your Name <your-email@nvidia.com>"
  tags:
    - isaac-os
---
```

- `name`: 1-64 characters, lowercase letters, digits, and single hyphens.
- `description`: 1-1024 characters stating the capability and when to use it, specific enough to
  distinguish the skill from nearby skills.
- `license`: required; use `Apache-2.0` to match the repository.
- `metadata.author`: required, in `Name <email@nvidia.com>` form.
- Put `version` (strict `x.y.z`), `tags`, and other custom fields under `metadata`, not at the top
  level. Do not use rule-only fields such as `globs` or `alwaysApply`.
- Add agent-specific fields such as `argument-hint` or `disable-model-invocation` only when the
  skill needs them, and keep any finding the validator reports on them visible.

## Content and security

- Give actionable, imperative steps with complete, runnable examples.
- Include no credentials, tokens, internal hostnames or IP addresses, or personal data other than
  the `author` field. Use placeholders in examples.
- Include no text that tries to override the agent's instructions, persona, or safety behavior,
  and no hidden or zero-width Unicode characters.
- Scripts must document usage, arguments, and exit codes, validate their inputs, avoid
  `shell=True` and shell strings built from variable input, keep writes inside their target
  directory, and use HTTPS with timeouts for network calls. Shell scripts start with
  `set -euo pipefail`.
- Use UTF-8 and end every text file with a newline.

## Product skills

End each product skill with a `## Links` section. It tells agents how to resolve the skill's
checkout-relative links without a checkout. Keep it aligned with **Find Isaac OS** in
`docs/for_agents.mdx`:

````md
## Links

The links in this skill are paths in an Isaac OS checkout. Without a checkout, find
docs pages in [llms.txt](https://docs.nvidia.com/isaac-os/llms.txt) for your Isaac OS
version, and other files in the
[source repository](https://github.com/nvidia-isaac/isaac-os) at your tag or commit.
If your version is not in `llms.txt`, read `docs/` in the source repository.
````

- Keep bundled references inside the skill directory. Other skills are optional: when you link
  one, also link its guide.
- Keep procedures and contracts in documentation. `docs/for_agents.mdx` holds the shared
  agent workflow, and the `isaac-os` skill links to it. Other product skills hold only their
  task workflow; link rather than duplicate.

## Local validation

SkillEvaluator checks a skill in three tiers:

- Tier 1: deterministic schema, security, PII, license, code-integrity, Unicode, quality, and lint
  checks.
- Tier 2: semantic overlap within the skill and with other skills.
- Tier 3: live agent tasks run with and without the skill to measure its effect. Tier 3 runs in
  the publication pipeline, not locally.

Install the pinned public release and the scanners Tier 1 calls. Record
`skillevaluator --version` with the results.

```bash
uv tool install --python 3.13 \
  "skillevaluator[all] @ git+https://github.com/NVIDIA/SkillEvaluator.git@v0.5.0"
uv tool install "git+https://github.com/NVIDIA/SkillSpector.git@v2.12.0"
# Also install gitleaks: https://github.com/gitleaks/gitleaks#installing
```

Run from the repository root; use `.agents/skills/<name>` for contributor skills:

```bash
# Validation 1: Tier 1.
skillevaluator validate skills/<name> --tiers 1 --no-llm

# Validation 2: Tier 1 and Tier 2.
skillevaluator validate skills/<name> --tiers 1,2
```

- Run validation 1 after every change to a skill and fix every failure it reports.
- Run validation 2 before requesting review. Tier 2 needs an embedding credential:
  `NVIDIA_API_KEY` by default, or the key for the provider set in
  `SKILL_EVAL_EMBEDDING_PROVIDER`. Without it, Tier 2 is skipped.
- SkillSpector needs an LLM credential even with `--no-llm`; without one, Tier 1 reports the
  Security Scan as `INCOMPLETE`. See the SkillSpector documentation for supported providers.
- The public release validates against the strict `external` profile.

- Report each validation as passed, failed with its findings, or not run with the reason. Report
  an `INCOMPLETE` or skipped check as not run, never as passing.

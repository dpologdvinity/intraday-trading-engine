# Agent skills

Process skills used by coding agents in this repository. Codex reads them from
`.agents/skills/`; `.claude/skills/` holds symlinks to the same directories.

| Skill | Use it for |
| --- | --- |
| `investigate-first` | Ambiguous failures: rank hypotheses by evidence before editing. |
| `surgical-patch` | Bug fixes at the narrowest responsible layer, with a regression test. |
| `safe-refactor` | Structural changes bracketed by verification, behavior preserved. |
| `lean-build` | Feature work with a strict scope and an explicit stop condition. |
| `verify-and-stop` | Validation-only tasks: the smallest proof set, then stop. |

## Attribution

These skills are unmodified copies from [Caveman](https://github.com/JuliusBrussee/caveman)
3.0.0, Copyright 2026 Julius Brussee, licensed under the Apache License 2.0
(see [LICENSE](LICENSE)). The rest of this repository is MIT-licensed.

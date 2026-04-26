# Multi-Agent Manuscript Review Setup

This folder contains a Claude Code project-local multi-agent review pipeline for the IEEE Internet of Things Journal manuscript.

## Folder layout

- `.claude/agents/structure-checker.md` — structure and IEEE format compliance audit.
- `.claude/agents/content-reviewer.md` — scientific content and methodology review.
- `.claude/agents/language-editor.md` — academic English and style review.
- `.claude/agents/reference-auditor.md` — BibTeX and citation integrity audit.
- `.claude/agents/final-qa.md` — final pre-submission consistency and readiness check.
- `.claude/commands/paper-review.md` — slash command to run the full pipeline.
- `review-logs/` — generated reports.
- `journal-guidelines/` — put IEEE IoT Journal author guidelines, template notes, or downloaded instructions here.
- `supplementary/related-papers/` — put 2-3 closely related PDF/TEX/notes files here for the content reviewer.
- `Parallel_Computing_on_FPGA/` — manuscript source.

## Before running a review

1. Open Claude Code from this folder or switch into it:
   ```bash
   cd D:/PROJECTS/Parallel_Computing_on_FPGA/Paper
   claude
   ```

2. Add journal guidance files to:
   ```text
   journal-guidelines/
   ```

3. Add 2-3 related papers or notes to:
   ```text
   supplementary/related-papers/
   ```

If these folders are empty, the agents will still run, but they will mark affected items as `VERIFY WITH JOURNAL` or note missing context.

## Run the full pipeline

Inside Claude Code, type:

```text
/paper-review 1
```

This generates:

```text
review-logs/round1-structure.md
review-logs/round1-content.md
review-logs/round1-language.md
review-logs/round1-references.md
review-logs/round1-final-qa.md
```

For the next revision round, run:

```text
/paper-review 2
```

## Run one agent manually

You can also ask Claude Code directly, for example:

```text
Use the structure-checker agent for review round 1 and write review-logs/round1-structure.md.
```

Or:

```text
Use the language-editor agent for review round 2 after my latest manuscript edits.
```

## Recommended workflow

1. Run `/paper-review 1` on the current draft.
2. Fix only CRITICAL and MAJOR issues first.
3. Re-run `/paper-review 2`.
4. Repeat until `final-qa` says `READY` or only manual verification items remain.
5. Apply MINOR/SUGGESTION items only if they improve clarity without destabilizing the manuscript.

## Important notes

- Agents are configured to audit and report only; they should not modify the manuscript.
- Reports require exact line numbers or section/table/figure references where possible.
- Each finding should include Severity, Location, Issue, Why it matters, and Suggested fix.
- The `final-qa` agent depends on the previous reports for the same round, so run it last.

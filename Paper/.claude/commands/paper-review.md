---
description: Run the IEEE IoT Journal manuscript review pipeline for a specified round.
argument-hint: "[round number, e.g. 1]"
allowed-tools: Task, Read, Glob, Grep, Bash
---

Run the manuscript review pipeline for round `$ARGUMENTS`.

If `$ARGUMENTS` is empty, use round `1`.

Work from the `Paper` directory. Use the manuscript in `Parallel_Computing_on_FPGA/` and write reports to `review-logs/`.

Run agents in this exact order, waiting for each report before starting the next:

1. `structure-checker` -> `review-logs/roundN-structure.md`
2. `content-reviewer` -> `review-logs/roundN-content.md`
3. `language-editor` -> `review-logs/roundN-language.md`
4. `reference-auditor` -> `review-logs/roundN-references.md`
5. `final-qa` -> `review-logs/roundN-final-qa.md`

For each agent, pass this instruction:

"Review round N. Work from the Paper directory. Read the manuscript at `Parallel_Computing_on_FPGA/main.tex` and its included files. Follow your role instructions exactly. Write your report to the specified `review-logs/roundN-*.md` file. Every finding must include Severity, Location, Issue, Why it matters, and Suggested fix. Use exact line numbers where possible. Do not modify the manuscript."

After all agents finish, summarize the generated report files and the top 5 blocking issues from `roundN-final-qa.md`.

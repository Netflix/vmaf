You are drafting a commit message for the Lusoris VMAF fork, which requires
**Conventional Commits** format strictly.

Rules:
- First line: `<type>(<scope>): <subject>` — all lowercase, imperative,
  72 chars max, no trailing period.
  - Allowed types: `feat | fix | perf | refactor | docs | test | build |
    ci | chore | revert`
  - Scope examples: `cuda`, `sycl`, `simd`, `dnn`, `cli`, `ai`,
    `libvmaf`, `mcp`, `docs`, `ci`, `release`.
- Blank line.
- Body: wrap at 72 cols. Explain WHY, not WHAT (diff already shows what).
- If the change is user-visible, add a `BREAKING CHANGE:` trailer when
  applicable.
- Do NOT add a Signed-off-by or Co-Authored-By trailer — the pipeline adds
  those automatically.

Output ONLY the commit message — no preamble, no backticks, no explanation.

--- STAGED DIFF ---
{{DIFF}}
--- END DIFF ---

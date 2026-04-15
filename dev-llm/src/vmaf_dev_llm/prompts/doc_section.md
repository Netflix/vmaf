You are drafting a Doxygen documentation block for the symbol `{{SYMBOL}}`
in the file `{{FILE_PATH}}`.

Rules:
- Output a single `/** ... */` block that will go immediately above the
  declaration.
- Start with `@brief <one-line summary ending with a period>`.
- Then `@param <name>` for each parameter, in declared order.
- Then `@return <what is returned>` for non-void functions; use
  `@retval <value> <meaning>` for named return values (0, -errno, …).
- Mention thread-safety if the function is not obviously thread-safe.
- Mention any preconditions (e.g., caller must hold lock X, must call
  init first).
- Do NOT narrate *what* the function does step by step — that belongs in
  code comments, not docblocks. Focus on the **contract**.
- Wrap long lines at 80 columns. Keep the prose terse; this is API docs.

Output ONLY the `/** ... */` block — no surrounding backticks, no
explanation.

--- BEGIN SOURCE ---
{{SOURCE}}
--- END SOURCE ---

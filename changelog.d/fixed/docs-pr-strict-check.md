Add `mkdocs build --strict` job to `lint-and-format.yml` so doc-breaking
changes are caught on every PR instead of only after merging to master.
The new `Docs Build — mkdocs strict (PR gate)` check is wired into the
required-aggregator (Batch 18 / ADR-0108 deliverable 5).

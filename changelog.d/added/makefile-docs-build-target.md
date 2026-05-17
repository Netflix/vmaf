- **build**: add `make docs-build` (`mkdocs build --strict`) and `make docs-serve`
  targets as local equivalents of the CI `docs.yml` required-check gate; closes
  the "CI-only gate with no local equivalent" finding from the 2026-05-16 test
  coverage audit.

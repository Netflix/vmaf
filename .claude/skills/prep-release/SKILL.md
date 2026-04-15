---
name: prep-release
description: Dry-run release-please locally, preview the CHANGELOG diff, verify signing prerequisites (Sigstore/OIDC), and surface any blockers before a release PR merges.
---

# /prep-release

## Invocation

```
/prep-release [--next-version=auto|<semver>]
```

## Steps

1. Check prerequisites:
   - `gh auth status` — logged in, has `workflow` scope.
   - `cosign version` — ≥ 2.2 (keyless requires modern cosign).
   - GitHub repo has `id-token: write` OIDC permission in `.github/workflows/release.yml`.
2. Run `release-please manifest-pr --dry-run --repo-url=<fork> --token=...` (via the
   release-please CLI or a scripted equivalent).
3. Parse the proposed release: version bump, CHANGELOG delta, affected packages.
4. Display the diff: version old → new, CHANGELOG section added, tag that will be
   created.
5. Verify the version matches our scheme: `v3.x.y-lusoris.N` (D11).
6. Report supply-chain prerequisites:
   - SBOM generator present (`syft`, `cyclonedx-cli`).
   - SLSA generator workflow configured.
   - Container image build target present (if applicable).
7. Summary: GO / NO-GO + blocker list.

## Notes

- This skill never creates releases. It only previews what the next release-please PR
  would propose, so the operator can merge it with confidence.
- If the proposed version doesn't match `v3.x.y-lusoris.N`, surface as a blocker; the
  fix is editing `release-please-config.json` (custom version pattern).

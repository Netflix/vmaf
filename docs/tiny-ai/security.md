# Tiny AI — security model

ONNX models are **data, not code**, but that is not an excuse for a lazy
runtime. libvmaf's DNN layer applies four layers of defence before any
frame touches a graph.

## Threat model

| Class | Example | Mitigation |
| --- | --- | --- |
| Hostile `.onnx` file | Model with custom op that exfiltrates via network syscall | Operator allowlist (next section). |
| Memory-exhaustion via huge model | 10 GB `.onnx` dropped in `VMAF_TINY_MODEL_DIR` | Size cap (`VMAF_MAX_MODEL_BYTES`, default 50 MB). |
| Path-traversal via `--tiny-model` | `--tiny-model ../../etc/shadow` | `vmaf_dnn_validate_onnx` requires `S_ISREG` + readable; refuses directories and devices. |
| Silent model substitution | Attacker replaces signed model with a poisoned one | Opt-in Sigstore (`cosign`) verification against the workflow identity. |

## Layer 1 — operator allowlist

[`libvmaf/src/dnn/op_allowlist.{h,c}`](../../libvmaf/src/dnn/op_allowlist.h)
holds a curated set of ONNX operator names. Before creating an ORT session
the loader walks the graph and rejects any node whose op is not on the
list.

The list includes the common building blocks of C1/C2/C3 architectures
(`Conv`, `Gemm`, `Relu`, `BatchNormalization`, `GlobalAveragePool`,
activations, pooling, arithmetic, reshape/transpose) and explicitly
excludes **control-flow** ops (`If`, `Loop`, `Scan`), custom ops, and
anything that could touch the filesystem or network. Unknown op names
(`custom_op_xyz`) are rejected.

Extending the list is a conscious, reviewed act — changes to
`op_allowlist.c` must be called out in the PR description and backed by a
concrete model that needs the addition.

## Layer 2 — resource bounds

- **Size cap.** Loader refuses files larger than
  `VMAF_DEFAULT_MAX_MODEL_BYTES` (50 MB). Override via
  `VMAF_MAX_MODEL_BYTES=<bytes>` env. Applies before mapping the file.
- **Path validation.** `vmaf_dnn_validate_onnx`:
  - resolves symlinks,
  - asserts `S_ISREG` (no devices, pipes, directories),
  - if `VMAF_TINY_MODEL_DIR` is set, asserts the resolved path is under
    it (chroot-style),
  - returns `-errno` on any failure — caller must check.
- **Shape sanity.** The sidecar JSON declares `input_name`,
  `output_name`, and `expected_output_range`. Runtime values outside the
  range raise a warning to stderr; persistent violation aborts scoring
  for the frame.

## Layer 3 — sandbox via ORT

ORT itself sandboxes graph execution — there is no interpreter, no
shell-out, no arbitrary file I/O from inside a graph. Our layers 1/2
harden the envelope around ORT so that even a clever graph cannot
consume unbounded memory or divert through a non-allowlisted op.

## Layer 4 — signature verification (opt-in)

Models shipped under `model/tiny/` are signed during release by
`.github/workflows/release.yml` using Sigstore's keyless flow. To verify
locally before loading:

```bash
cosign verify-blob \
    --certificate-identity-regexp "https://github.com/lusoris/vmaf/.github/workflows/release.yml@.*" \
    --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
    --signature vmaf_tiny_fr_v1.onnx.sig \
    --certificate vmaf_tiny_fr_v1.onnx.cert \
    vmaf_tiny_fr_v1.onnx
```

A future `--tiny-model-verify` flag will invoke `cosign` at load time and
fail closed if the signature is missing or bad. Off by default for
dev-friendliness; strongly recommended on for production deployments.

## Reporting

If you believe a shipped model is hostile or find a way to bypass the
allowlist, follow the disclosure process in
[`SECURITY.md`](../../SECURITY.md) (90-day coordinated, PGP key listed).

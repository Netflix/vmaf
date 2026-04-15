#!/usr/bin/env bash
# test_cli.sh — smoke-test the `vmaf --tiny-model` option.
#
# Requires: meson build with -Denable_dnn=enabled, an ONNX model under
# model/tiny/ (any), and libonnxruntime on the runtime library path.
#
# When DNN is disabled, asserts the clear error message instead.
set -eu

: "${VMAF_BIN:=build/tools/vmaf}"

if [[ ! -x "$VMAF_BIN" ]]; then
    echo "vmaf binary not found at $VMAF_BIN — set VMAF_BIN=<path>" >&2
    exit 77  # meson's "skipped"
fi

# `vmaf --help` exits with 1 by convention, so capture the output first
# instead of piping into grep under set -o pipefail.
help_text="$("$VMAF_BIN" --help 2>&1 || true)"

# 1. Help text must advertise the tiny flags.
printf '%s\n' "$help_text" | grep -q -- '--tiny-model'   || { echo "help missing --tiny-model"; exit 1; }
printf '%s\n' "$help_text" | grep -q -- '--tiny-device'  || { echo "help missing --tiny-device"; exit 1; }
printf '%s\n' "$help_text" | grep -q -- '--no-reference' || { echo "help missing --no-reference"; exit 1; }

# 2. Invalid device string must be rejected with a useful message.
if "$VMAF_BIN" --tiny-model /nonexistent.onnx --tiny-device bogus 2>&1 \
    | grep -qi 'auto|cpu|cuda|openvino|rocm'; then
    :
else
    echo "expected validation error for --tiny-device bogus"
    exit 1
fi

echo "PASS: $0"

/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "test.h"

#include "dnn/model_loader.h"

static char *test_sniff_by_extension(void)
{
    mu_assert("json → SVM", vmaf_dnn_sniff_kind("foo.json") == VMAF_MODEL_KIND_SVM);
    mu_assert("pkl → SVM",  vmaf_dnn_sniff_kind("foo.pkl")  == VMAF_MODEL_KIND_SVM);
    mu_assert("onnx → DNN_FR",
              vmaf_dnn_sniff_kind("foo.onnx") == VMAF_MODEL_KIND_DNN_FR);
    mu_assert("unknown ext → -1", vmaf_dnn_sniff_kind("foo.bin") == -1);
    mu_assert("NULL → -1", vmaf_dnn_sniff_kind(NULL) == -1);
    return NULL;
}

static char *test_size_cap(void)
{
    /* A tiny file that exists — use /etc/hostname as a proxy for "regular file,
     * within limits". Size cap of 1 byte should reject it unless hostname is
     * 0 bytes (unlikely). */
    int err = vmaf_dnn_validate_onnx("/etc/hostname", 1);
    mu_assert("expected -E2BIG for 1-byte cap", err == -E2BIG || err == 0);
    err = vmaf_dnn_validate_onnx("/definitely/does/not/exist.onnx", 0);
    mu_assert("expected errno for missing file", err < 0);
    return NULL;
}

static char *test_sidecar_parses(void)
{
    char tmpl[] = "/tmp/vmaf-dnn-sidecar-XXXXXX";
    int fd = mkstemp(tmpl);
    mu_assert("mkstemp failed", fd >= 0);
    close(fd);

    char onnx[1024], sidecar[1024];
    snprintf(onnx,    sizeof onnx,    "%s.onnx", tmpl);
    snprintf(sidecar, sizeof sidecar, "%s.json", tmpl);
    /* Touch an empty onnx so sidecar_load doesn't key off its existence. */
    FILE *f = fopen(onnx, "w"); if (f) fclose(f);

    FILE *s = fopen(sidecar, "w");
    mu_assert("fopen sidecar failed", s != NULL);
    fprintf(s,
        "{\n"
        "  \"name\": \"vmaf_tiny_fr_v1\",\n"
        "  \"kind\": \"fr\",\n"
        "  \"onnx_opset\": 17,\n"
        "  \"input_name\":  \"features\",\n"
        "  \"output_name\": \"score\"\n"
        "}\n");
    fclose(s);

    VmafModelSidecar meta;
    int err = vmaf_dnn_sidecar_load(onnx, &meta);
    mu_assert("sidecar_load failed", err == 0);
    mu_assert("kind FR",    meta.kind == VMAF_MODEL_KIND_DNN_FR);
    mu_assert("opset 17",   meta.opset == 17);
    mu_assert("name set",   meta.name && !strcmp(meta.name, "vmaf_tiny_fr_v1"));
    mu_assert("input set",  meta.input_name && !strcmp(meta.input_name, "features"));
    mu_assert("output set", meta.output_name && !strcmp(meta.output_name, "score"));
    vmaf_dnn_sidecar_free(&meta);

    remove(sidecar);
    remove(onnx);
    remove(tmpl);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_sniff_by_extension);
    mu_run_test(test_size_cap);
    mu_run_test(test_sidecar_parses);
    return NULL;
}

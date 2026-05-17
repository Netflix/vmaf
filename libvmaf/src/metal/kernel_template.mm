/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Metal kernel-template helper bodies — runtime implementation
 *  (T8-1b / ADR-0420). Replaces the C scaffold's -ENOSYS stubs with
 *  real `[id<MTLDevice> newCommandQueue]` +
 *  `[id<MTLDevice> newSharedEvent]` +
 *  `[id<MTLBlitCommandEncoder> fillBuffer:range:value:]` +
 *  `[id<MTLCommandBuffer> waitUntilCompleted]` plumbing.
 *
 *  Handle ABI: the public-facing `kernel_template.h` carries
 *  `uintptr_t` slots so the header stays free of `<Metal/Metal.h>`.
 *  This TU bridges `uintptr_t` ↔ id<MTL...> via __bridge_retained /
 *  __bridge_transfer; the C struct owns +1 retains for the duration
 *  of init→close. Mirrors the MetalCpp pattern documented in
 *  ADR-0361 §"Header purity".
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

extern "C" {
#include "../kernel_lifecycle_common.h"
#include "common.h"
#include "kernel_template.h"
}

/*
 * Bridge helpers. The struct stores `uintptr_t`; the .mm TU bridges
 * back and forth via void * intermediates. `_retain` consumes a +1
 * ARC reference from the autoreleased Metal handle and stashes it in
 * the uintptr_t slot; `_transfer` reverses that to release.
 */
static inline uintptr_t retain_as_uptr(id obj)
{
    if (obj == nil) {
        return 0;
    }
    return (uintptr_t)(__bridge_retained void *)obj;
}

static inline id transfer_from_uptr(uintptr_t slot)
{
    if (slot == 0) {
        return nil;
    }
    void *p = (void *)slot;
    return (__bridge_transfer id)p;
}

static inline id borrow_from_uptr(uintptr_t slot)
{
    if (slot == 0) {
        return nil;
    }
    return (__bridge id)(void *)slot;
}

int vmaf_metal_kernel_lifecycle_init(VmafMetalKernelLifecycle *lc, VmafMetalContext *ctx)
{
    if (lc == NULL) {
        return -EINVAL;
    }
    VMAF_LIFECYCLE_ZERO(lc);

    if (ctx == NULL) {
        return -EINVAL;
    }

    void *device_handle = vmaf_metal_context_device_handle(ctx);
    if (device_handle == NULL) {
        return -ENODEV;
    }
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_handle;

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
        return -ENOMEM;
    }
    id<MTLSharedEvent> submit_ev = [device newSharedEvent];
    if (submit_ev == nil) {
        return -ENOMEM;
    }
    id<MTLSharedEvent> finished_ev = [device newSharedEvent];
    if (finished_ev == nil) {
        return -ENOMEM;
    }

    lc->cmd_queue = retain_as_uptr(queue);
    lc->submit    = retain_as_uptr(submit_ev);
    lc->finished  = retain_as_uptr(finished_ev);

    return 0;
}

int vmaf_metal_kernel_buffer_alloc(VmafMetalKernelBuffer *buf, VmafMetalContext *ctx, size_t bytes)
{
    if (buf == NULL) {
        return -EINVAL;
    }
    VMAF_LIFECYCLE_ZERO(buf);
    buf->bytes = bytes;

    if (ctx == NULL) {
        return -EINVAL;
    }
    if (bytes == 0) {
        return -EINVAL;
    }

    void *device_handle = vmaf_metal_context_device_handle(ctx);
    if (device_handle == NULL) {
        return -ENODEV;
    }
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_handle;

    id<MTLBuffer> b = [device newBufferWithLength:bytes
                                          options:MTLResourceStorageModeShared];
    if (b == nil) {
        return -ENOMEM;
    }
    buf->buffer    = retain_as_uptr(b);
    buf->host_view = [b contents];
    return 0;
}

int vmaf_metal_kernel_submit_pre_launch(VmafMetalKernelLifecycle *lc, VmafMetalContext *ctx,
                                        VmafMetalKernelBuffer *buf,
                                        uintptr_t picture_command_buffer,
                                        uintptr_t dist_ready_event)
{
    (void)ctx;
    if (lc == NULL || buf == NULL) {
        return -EINVAL;
    }
    if (lc->cmd_queue == 0 || buf->buffer == 0) {
        return -EINVAL;
    }

    id<MTLCommandQueue> queue = (id<MTLCommandQueue>)borrow_from_uptr(lc->cmd_queue);
    id<MTLBuffer> accum       = (id<MTLBuffer>)borrow_from_uptr(buf->buffer);

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil) {
        return -ENOMEM;
    }

    /* Step 1: zero the accumulator on our private command queue. */
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    if (blit == nil) {
        return -ENOMEM;
    }
    [blit fillBuffer:accum range:NSMakeRange(0, buf->bytes) value:0];
    [blit endEncoding];

    /* Step 2: wait for the dist-side ready event on the picture's
     * command buffer (caller-provided, so the picture work and our
     * pre-launch are causally ordered). When either handle is 0 we
     * skip the cross-queue wait — this is the path during single-
     * buffer testing where there's no producer queue to sync with. */
    if (picture_command_buffer != 0 && dist_ready_event != 0) {
        id<MTLCommandBuffer> pic_cmd =
            (id<MTLCommandBuffer>)borrow_from_uptr(picture_command_buffer);
        id<MTLEvent> evt = (id<MTLEvent>)borrow_from_uptr(dist_ready_event);
        [pic_cmd encodeWaitForEvent:evt value:1];
    }

    [cmd commit];
    return 0;
}

int vmaf_metal_kernel_collect_wait(VmafMetalKernelLifecycle *lc, VmafMetalContext *ctx)
{
    (void)ctx;
    if (lc == NULL) {
        return -EINVAL;
    }
    if (lc->cmd_queue == 0) {
        return -EINVAL;
    }

    /* Drain the private command queue. The simplest correct sequence
     * commits a no-op blit + waits on its command buffer; this fences
     * everything previously enqueued, which is what consumers expect
     * after submit() before reading [contents] in collect(). */
    id<MTLCommandQueue> queue = (id<MTLCommandQueue>)borrow_from_uptr(lc->cmd_queue);
    id<MTLCommandBuffer> fence = [queue commandBuffer];
    if (fence == nil) {
        return -ENOMEM;
    }
    [fence commit];
    [fence waitUntilCompleted];
    return 0;
}

int vmaf_metal_kernel_lifecycle_close(VmafMetalKernelLifecycle *lc, VmafMetalContext *ctx)
{
    (void)ctx;
    if (lc == NULL) {
        return 0;
    }

    /* Best-effort drain before release so any in-flight work finishes. */
    if (lc->cmd_queue != 0) {
        id<MTLCommandQueue> queue = (id<MTLCommandQueue>)borrow_from_uptr(lc->cmd_queue);
        id<MTLCommandBuffer> fence = [queue commandBuffer];
        if (fence != nil) {
            [fence commit];
            [fence waitUntilCompleted];
        }
    }

    /* Bridge-transfer each slot back to ARC ownership so the +1
     * retains we took in init() are released by scope exit. */
    if (lc->finished != 0) {
        id ev __attribute__((unused)) = transfer_from_uptr(lc->finished);
        lc->finished = 0;
    }
    if (lc->submit != 0) {
        id ev __attribute__((unused)) = transfer_from_uptr(lc->submit);
        lc->submit = 0;
    }
    if (lc->cmd_queue != 0) {
        id q __attribute__((unused)) = transfer_from_uptr(lc->cmd_queue);
        lc->cmd_queue = 0;
    }
    return 0;
}

int vmaf_metal_kernel_buffer_free(VmafMetalKernelBuffer *buf, VmafMetalContext *ctx)
{
    (void)ctx;
    if (buf == NULL) {
        return 0;
    }
    if (buf->buffer != 0) {
        id b __attribute__((unused)) = transfer_from_uptr(buf->buffer);
        buf->buffer = 0;
    }
    buf->host_view = NULL;
    buf->bytes     = 0;
    return 0;
}

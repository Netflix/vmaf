#pragma once

#ifndef CPUDETECT_H_
#define CPUDETECT_H_

#if defined(_MSC_VER)
  #include <intrin.h>
#elif defined(__GNUC__)
  #include <cpuid.h>
#endif

/**
 * Bitfield of selected x86 feature flags.
 */
typedef struct X86Capabilities {
    unsigned sse   : 1;
    unsigned sse2  : 1;
    unsigned sse3  : 1;
    unsigned ssse3 : 1;
    unsigned fma   : 1;
    unsigned sse41 : 1;
    unsigned sse42 : 1;
    unsigned avx   : 1;
    unsigned f16c  : 1;
    unsigned avx2  : 1;
} X86Capabilities;

/**
 * Execute the CPUID instruction.
 *
 * @param regs array to receive eax, ebx, ecx, edx
 * @param eax argument to instruction
 * @param ecx argument to instruction
 */
void do_cpuid(int regs[4], int eax, int ecx)
{
#if defined(_MSC_VER)
	__cpuidex(regs, eax, ecx);
#elif defined(__GNUC__)
	__cpuid_count(eax, ecx, regs[0], regs[1], regs[2], regs[3]);
#else
	regs[0] = 0;
	regs[1] = 0;
	regs[2] = 0;
	regs[3] = 0;
#endif
}

/**
 * Get the x86 feature flags on the current CPU.
 *
 * @return capabilities
 */
X86Capabilities query_x86_capabilities()
{
	X86Capabilities caps = { 0 };
	int regs[4] = { 0 };

	do_cpuid(regs, 1, 0);
	caps.sse   = !!(regs[3] & (1 << 25));
	caps.sse2  = !!(regs[3] & (1 << 26));
	caps.sse3  = !!(regs[2] & (1 << 0));
	caps.ssse3 = !!(regs[2] & (1 << 9));
	caps.fma   = !!(regs[2] & (1 << 12));
	caps.sse41 = !!(regs[2] & (1 << 19));
	caps.sse42 = !!(regs[2] & (1 << 20));
	caps.avx   = !!(regs[2] & (1 << 28));
	caps.f16c  = !!(regs[2] & (1 << 29));

	do_cpuid(regs, 7, 0);
	caps.avx2 = !!(regs[1] & (1 << 5));

	return caps;
}

#endif // CPUDETECT_H_

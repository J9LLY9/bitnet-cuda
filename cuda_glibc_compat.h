// Workaround for CUDA 13.1 + glibc 2.41+ (GCC 15) rsqrt conflict.
// glibc declares rsqrt with noexcept(true), CUDA declares it without.
// We suppress glibc's rsqrt/rsqrtf by hiding the IEC 60559 C23 section.
#pragma once

// Force-redefine the guard that controls glibc's rsqrt declarations.
// The glibc header does #undef then #define, so a -D flag won't stick.
// Instead we intercept __GLIBC_USE by redefining it for this specific feature.
#include <features.h>

#ifdef __GLIBC_USE_IEC_60559_FUNCS_EXT_C23
#undef __GLIBC_USE_IEC_60559_FUNCS_EXT_C23
#define __GLIBC_USE_IEC_60559_FUNCS_EXT_C23 0
#define _BITNET_PATCHED_IEC_60559 1
#endif

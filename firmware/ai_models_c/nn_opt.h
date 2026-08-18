/******************************************************************************
 * nn_opt.h  —  force speed optimization on the AI math translation units.
 *
 * The Keil project default is -O0 (debug build).  At -O0 armcc spills every
 * loop variable to the stack and does no FPU scheduling, so the inner MAC loop
 * of nn_linear/attention costs ~15-30 cycles/iter -> AI-3 forward measured
 * 598 ms on-chip (DWT @600 MHz).  These engines are pure float32 math with no
 * timing-sensitive delays and no undefined behaviour (host-verified clean at
 * clang -O2 -Wall -Wextra), so it is safe to compile *only these files* at
 * -O3 -Otime while leaving the timing-sensitive driver code (soft-I2C bit-bang,
 * LCD, sensors) at the project default -O0.
 *
 * armcc (ARM Compiler 5) honours per-file `#pragma Onum` / `#pragma Otime`;
 * the guard makes the pragma a no-op under armclang (AC6) and host clang, where
 * __ARMCC_VERSION is either >= 6000000 or undefined.
 *
 * USAGE: #include "nn_opt.h" AFTER all other #includes, before the first
 *        function definition, in each AI engine .c file.
 ******************************************************************************/
#ifndef NN_OPT_H
#define NN_OPT_H

#if defined(__ARMCC_VERSION) && (__ARMCC_VERSION < 6000000)
#pragma O3
#pragma Otime
#endif

#endif /* NN_OPT_H */

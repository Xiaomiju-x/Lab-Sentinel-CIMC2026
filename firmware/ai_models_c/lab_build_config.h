/******************************************************************************
 * lab_build_config.h — one-line build switches for fast iteration.
 *
 * LAB_LM_ENABLE — the on-chip GENERATIVE LM stack:
 *     • flagship nano-LM x1p9  (1.8 MB INT8 weights, internal Flash)
 *     • 5-expert LLM cluster   (SPI-flash swap-load)
 *     • 3-size LM bank         (SPI-flash swap-load: m1p35 / s0p6)
 *     • online-learning head   (runs in nlm_task)
 *
 *   DEFINED  (default, production): the real generative LM is built + run.
 *   UNDEFINED (comment the line out, UI-dev FAST build):
 *     - the 1.8 MB flagship INT8 weights are EXCLUDED from the firmware image
 *       (ai_nanolm.c) -> image ~1.8 MB smaller -> flash + reboot ~2x faster
 *     - the NLM / OL / CLUSTER / BANK boot self-tests are skipped (lab_sentinel.c)
 *     - nlm_task / cluster_task are not spawned
 *   The HMI shows an "LM disabled (UI-dev build)" placeholder; the 20 discriminative
 *   AI models, controller, sensors and the whole UI are unaffected.
 *
 *   >>> To do fast UI work: COMMENT OUT the next line, rebuild, flash. <<<
 *   >>> When done with the UI:  UNCOMMENT it, rebuild, flash, re-provision. <<<
 *
 * (NLM_HOST_TEST builds always compile the full engine regardless of this switch.)
 ******************************************************************************/
#ifndef LAB_BUILD_CONFIG_H
#define LAB_BUILD_CONFIG_H

#define LAB_LM_ENABLE 1         /* ENABLED = full production build (flagship x1p9 + 7-expert cluster + 3-size bank).
                                 * ENABLED 2026-07-05 for video recording: shows the full 30-model edge-AI stack.
                                 * SPI-flash blobs persist -> rebuild + flash should bring the generative LM back
                                 * without re-provisioning if the blobs are still present. */

/* LAB_AI_BOOT_SELFTEST — discriminative 20-model golden self-test + DWT latency
 * probe at boot (the [AI] selftest / maxerr / latency lines). DEFINED = run+print
 * (production). COMMENTED OUT = skip for fast bring-up (saves a few hundred ms of
 * boot; the 20 models are still compiled and still run at RUNTIME in vision/env/
 * fusion tasks — only the boot-time verification print is skipped). For fast
 * MAX31855 iteration: leave commented; re-enable once the TC module is verified. */
#define LAB_AI_BOOT_SELFTEST 1

#endif /* LAB_BUILD_CONFIG_H */

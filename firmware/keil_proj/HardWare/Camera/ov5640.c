/******************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 *
 * Public, camera-disabled OV5640 adapter.
 *
 * The competition firmware used a sensor register table whose redistribution
 * permission could not be established.  This clean-room adapter keeps the
 * public project link-complete and fails the camera probe explicitly.  Supply
 * a reviewed, permissively licensed implementation to enable live capture.
 ******************************************************************************/

#include "ov5640.h"
#include "sdram.h"

volatile uint16_t * const ov5640_framebuf =
    (volatile uint16_t * const)SDRAM_CAMERA_FB;

uint8_t ov5640_init(void)
{
    return 2U; /* feature unavailable: do not claim LIVE_SENSOR */
}

uint8_t ov5640_read_chip_id(uint16_t *id)
{
    if (id != 0) {
        *id = 0U;
    }
    return 1U;
}

uint8_t ov5640_frame_ready(void)
{
    return 0U;
}

void ov5640_frame_consume(void)
{
}

void ov5640_capture_one(void)
{
}

void ov5640_dvp_pads(uint8_t on)
{
    (void)on;
}

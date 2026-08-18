/******************************************************************************
 * lv_port_indev.c
 *
 * LVGL 8.3 pointer input driver — GT911 capacitive touch (soft I2C).
 *
 * Single-loop design: the LVGL read_cb scans the GT911 directly. It is called
 * by LVGL's indev read_timer inside lv_timer_handler() (ui_task), on the same
 * ~33 ms cadence as the render. A short hold-latch bridges the rare GT911 frame
 * gap so a single touch doesn't stutter.
 *
 * NOTE (2026-06-02): a dedicated high-priority touch_task polling the GT911 at
 * ~66 Hz was tried to make quick nav-tab taps land more reliably. It made things
 * WORSE — all LVGL work (render + button/tab event dispatch) runs in ui_task, so
 * a PRIO_HIGH sampler that busy-waits through the soft-I2C bit-delays starved
 * that loop -> the screen rendered AND processed input less often (worse flicker,
 * less responsive buttons). Reverted to this single-loop version. The correct
 * improvement is interrupt-driven touch via the GT911 INT pin (PH15) — a clean
 * separate change.
 ******************************************************************************/

#include "lv_port_indev.h"
#include "gt911.h"

static lv_indev_drv_t s_indev_drv;

/* Last raw touch coordinate + down flag — exported for the ui_task [touch] trace
 * (one line per tap, shows where taps land relative to the buttons). */
volatile int     g_touch_raw_x = 0;
volatile int     g_touch_raw_y = 0;
volatile uint8_t g_touch_down  = 0U;

/* Bridge a missed GT911 frame so a single touch doesn't stutter. The read_cb
 * runs at the render-loop rate (~33 ms), and the GT911's own ~10 ms frame rate
 * is far faster, so a real mid-touch gap almost never spans a whole read. hold=1
 * bridges the rare single-frame miss but releases ~1 read after lift, so a quick
 * second tap (MOTOR toggle / ABORT confirm) is NOT swallowed by a phantom press.
 * (Was 3, which at the slow loop rate held a phantom PRESSED ~500 ms and ate the
 * second tap.) */
#define TOUCH_HOLD_POLLS  1U

/* LVGL input read: scan the GT911 and report the latched state. */
static void _touch_read_cb(lv_indev_drv_t *drv, lv_indev_data_t *data)
{
    static uint8_t hold = 0U;
    gt911_point_t pts[1];
    uint8_t cnt = gt911_scan(pts, 1U);

    if (cnt > 0U) {
        g_touch_raw_x = (int)pts[0].x;
        g_touch_raw_y = (int)pts[0].y;
        g_touch_down  = 1U;
        hold = TOUCH_HOLD_POLLS;
    } else if (hold > 0U) {
        hold--;                  /* bridge a missed frame / brief hold after lift */
        g_touch_down = 1U;
    } else {
        g_touch_down = 0U;
    }

    data->state   = g_touch_down ? LV_INDEV_STATE_PRESSED : LV_INDEV_STATE_RELEASED;
    data->point.x = (lv_coord_t)g_touch_raw_x;
    data->point.y = (lv_coord_t)g_touch_raw_y;
    (void)drv;
}

void lv_port_indev_init(void)
{
    lv_indev_drv_init(&s_indev_drv);
    s_indev_drv.type    = LV_INDEV_TYPE_POINTER;
    s_indev_drv.read_cb = _touch_read_cb;
    lv_indev_drv_register(&s_indev_drv);
}

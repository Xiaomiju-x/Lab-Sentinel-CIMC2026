/******************************************************************************
 * Function.c
 *
 * CIMC Lab-Sentinel - System bootstrap (called from main.c).
 *   System_Init() : MCU/Cache/SysTick init.
 *   UsrFunction() : Hands off to FreeRTOS via app_main().
 *
 ******************************************************************************/

#include "Function.h"
#include "app_main.h"

static void cache_enable(void);

/******************************************************************************
 * System_Init
 *  - Enable I-Cache + D-Cache (Cortex-M7).
 *  - Configure SysTick for 1 ms tick (used as FreeRTOS tick source).
 ******************************************************************************/
void System_Init(void)
{
    cache_enable();
    systick_config();
}

/******************************************************************************
 * UsrFunction
 *  - Entry point post system init. Hands off to FreeRTOS scheduler.
 *  - app_main() never returns under normal operation.
 ******************************************************************************/
void UsrFunction(void)
{
    app_main();

    /* Should not reach here. If we do, scheduler returned (out of heap or
     * an unrecoverable fault). Loop forever for debugger to catch.            */
    for (;;) { }
}

static void cache_enable(void)
{
    SCB_EnableICache();
    SCB_EnableDCache();
}

/******************************* End of File *********************************/

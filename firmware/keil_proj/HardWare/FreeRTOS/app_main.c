/******************************************************************************
 * app_main.c
 *
 * Thin shim. Real FreeRTOS task framework lives in
 * HardWare/Lab_Sentinel/lab_sentinel.{c,h}.
 *
 ******************************************************************************/

#include "app_main.h"
#include "lab_sentinel.h"

void app_main(void)
{
    lab_sentinel_main();
}

/******************************* End of File *********************************/

/******************************************************************************
 * main.c
 *
 * CIMC Lab-Sentinel - 2026 CIMC "Siemens Cup" Edge AI Electronic System
 *
 * Boot flow:
 *   main()
 *    -> System_Init()  (cache + systick)
 *    -> UsrFunction()  -> app_main() -> lab_sentinel_main()
 *                                        -> FreeRTOS scheduler
 *
 ******************************************************************************/

#include "Function.h"

int main(void)
{
    System_Init();
    UsrFunction();    /* never returns */
}

/******************************* End of File *********************************/

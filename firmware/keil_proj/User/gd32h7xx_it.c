/*!
    \file    gd32f4xx_it.c
    \brief   interrupt service routines
    
    \version 2020-09-04, V2.0.0, demo for GD32F4xx
*/

/*
    Copyright (c) 2020, GigaDevice Semiconductor Inc.

    Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this 
       list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, 
       this list of conditions and the following disclaimer in the documentation 
       and/or other materials provided with the distribution.
    3. Neither the name of the copyright holder nor the names of its contributors 
       may be used to endorse or promote products derived from this software without 
       specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
OF SUCH DAMAGE.
*/

#include "gd32h7xx_it.h"
#include "gd32h7xx.h"      /* UART4 + usart_*, SCB fault regs, __get_PSP */
#include "systick.h"

/* ---- TEMP fault diagnostic: dump the stacked PC/LR + fault status regs over
 * UART4 (raw, no FreeRTOS) so a HardFault prints WHERE it faulted instead of
 * silently spinning until the watchdog resets. Map the PC against the .map. */
static void _flt_putc(char c)
{
    usart_data_transmit(UART4, (uint8_t)c);
    while (RESET == usart_flag_get(UART4, USART_FLAG_TBE)) { }
}
static void _flt_puts(const char *s) { while (*s) { _flt_putc(*s++); } }
static void _flt_hex(uint32_t v)
{
    static const char H[] = "0123456789ABCDEF";
    char b[9]; int i;
    b[8] = '\0';
    for (i = 7; i >= 0; i--) { b[i] = H[v & 0xFu]; v >>= 4; }
    _flt_puts(b);
}
static void _flt_dump(const char *tag)
{
    /* task faults stack onto PSP (FreeRTOS tasks use PSP) */
    uint32_t *sp = (uint32_t *)__get_PSP();
    _flt_puts("\r\n*** ");  _flt_puts(tag);     _flt_puts(" ***\r\n");
    _flt_puts("PC=");       _flt_hex(sp[6]);
    _flt_puts(" LR=");      _flt_hex(sp[5]);
    _flt_puts(" PSR=");     _flt_hex(sp[7]);
    _flt_puts("\r\nCFSR=");  _flt_hex(*(volatile uint32_t *)0xE000ED28u);
    _flt_puts(" HFSR=");     _flt_hex(*(volatile uint32_t *)0xE000ED2Cu);
    _flt_puts(" MMFAR=");    _flt_hex(*(volatile uint32_t *)0xE000ED34u);
    _flt_puts(" BFAR=");     _flt_hex(*(volatile uint32_t *)0xE000ED38u);
    _flt_puts("\r\n");
    while (1) { }
}

/*!
    \brief      this function handles NMI exception
    \param[in]  none
    \param[out] none
    \retval     none
*/
void NMI_Handler(void)
{
}

/*!
    \brief      this function handles HardFault exception
    \param[in]  none
    \param[out] none
    \retval     none
*/
void HardFault_Handler(void)
{
    _flt_dump("HARDFAULT");
}

/*!
    \brief      this function handles MemManage exception
    \param[in]  none
    \param[out] none
    \retval     none
*/
void MemManage_Handler(void)
{
    _flt_dump("MEMMANAGE");
}

/*!
    \brief      this function handles BusFault exception
    \param[in]  none
    \param[out] none
    \retval     none
*/
void BusFault_Handler(void)
{
    _flt_dump("BUSFAULT");
}

/*!
    \brief      this function handles UsageFault exception
    \param[in]  none
    \param[out] none
    \retval     none
*/
void UsageFault_Handler(void)
{
    _flt_dump("USAGEFAULT");
}

/*!
    \brief      this function handles SVC exception
    \param[in]  none
    \param[out] none
    \retval     none
*/
#if 0  /* FreeRTOS owns SVC_Handler (aliased via FreeRTOSConfig.h) */
void SVC_Handler(void)
{
}
#endif

/*!
    \brief      this function handles DebugMon exception
    \param[in]  none
    \param[out] none
    \retval     none
*/
void DebugMon_Handler(void)
{
}

/*!
    \brief      this function handles PendSV exception
    \param[in]  none
    \param[out] none
    \retval     none
*/
#if 0  /* FreeRTOS owns PendSV_Handler (aliased via FreeRTOSConfig.h) */
void PendSV_Handler(void)
{
}
#endif

/*!
    \brief      this function handles SysTick exception
    \param[in]  none
    \param[out] none
    \retval     none
*/
#if 0  /* FreeRTOS owns SysTick_Handler. delay_decrement() is now called from
        * vApplicationTickHook() in lab_sentinel.c so that GD32 systick.c
        * blocking delay_1ms() keeps working under the FreeRTOS scheduler. */
void SysTick_Handler(void)
{
    delay_decrement();
}
#endif

/******************************************************************************
 * inmp441.c — SPI0 I²S master receive + DMA, 16 kHz left-channel audio.
 *
 * Architecture:
 *   TIMER1 CH0 (PA0, AF1) generates WS at 16 kHz, 50 % duty.
 *   SPI0 acts as I²S0 master: outputs CK on PA5 (AF5), receives SD on PA7 (AF5).
 *   DMA0 CH0 (SPI0_RX request via DMAMUX) continuously fills s_buf in circular
 *   mode. Each 32-bit DMA word contains the 16-bit audio sample in the upper
 *   half (Philips I²S left-justified in a 32-bit frame).
 *
 * Pin rationale (2026-05-15):
 *   Old PD6 conflicts with CI1302 USART1_RX. New PA5/PA7 is on SPI0.
 *   PA7 also overlaps Ethernet RMII CRS_DV (AF11); Ethernet is intentionally
 *   disabled (Phase 6 optional) so PA7 is free for the mic.
 *   PA5 is also wired as SCCB SCL (sccb.c) but OV5640 is dead (PD0=SDRAM D2
 *   conflict, ov5640_init not called) so SCCB never configures PA5.
 *
 * Clock math (APB2 assumed 120 MHz, TIMER1/SPI0 both on APB2):
 *   BCLK = 16000 × 32 = 512 kHz
 *   SPI0 odd-prescaler: 120 MHz / (2 × 117 + 1) = ~512 kHz  → use I2SPSC
 *   TIMER1 period = 120 MHz / 16 kHz = 7500 ticks, CH0 compare = 3750 (50 %)
 ******************************************************************************/

#include "inmp441.h"

/* ---------- pin / peripheral map ---------- */
#define WS_PORT   GPIOA
#define WS_PIN    GPIO_PIN_0   /* TIMER1_CH0 AF1 */

#define CK_PORT   GPIOA
#define CK_PIN    GPIO_PIN_5   /* SPI0_SCK  AF5  */

#define SD_PORT   GPIOA
#define SD_PIN    GPIO_PIN_7   /* SPI0_MOSI AF5  (input in I²S receive mode) */

#define LR_PORT   GPIOD
#define LR_PIN    GPIO_PIN_5   /* GPIO output LOW = left channel */

/* ---------- DMA buffer (16-bit samples, circular) ---------- */
static volatile int16_t s_buf[INMP441_BUF_SAMPLES];
static volatile uint32_t s_write_idx = 0U;   /* updated by DMA half/full ISR */

/* ============================================================================
 * WS generator: TIMER1 CH0, 16 kHz 50 % duty on PA0 (AF1)
 * ============================================================================ */
static void _ws_timer_init(void)
{
    rcu_periph_clock_enable(RCU_TIMER1);
    rcu_periph_clock_enable(RCU_GPIOA);

    gpio_af_set(WS_PORT, GPIO_AF_1, WS_PIN);
    gpio_mode_set(WS_PORT, GPIO_MODE_AF, GPIO_PUPD_NONE, WS_PIN);
    gpio_output_options_set(WS_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_60MHZ, WS_PIN);

    timer_deinit(TIMER1);

    timer_parameter_struct tp;
    timer_struct_para_init(&tp);
    tp.prescaler         = 0U;
    tp.alignedmode       = TIMER_COUNTER_EDGE;
    tp.counterdirection  = TIMER_COUNTER_UP;
    tp.period            = 7499U;   /* 120 MHz / 7500 = 16 kHz */
    tp.clockdivision     = TIMER_CKDIV_DIV1;
    tp.repetitioncounter = 0U;
    timer_init(TIMER1, &tp);

    timer_oc_parameter_struct oc;
    timer_channel_output_struct_para_init(&oc);
    oc.outputstate  = TIMER_CCX_ENABLE;
    oc.ocpolarity   = TIMER_OC_POLARITY_HIGH;
    oc.ocidlestate  = TIMER_OC_IDLE_STATE_LOW;
    timer_channel_output_config(TIMER1, TIMER_CH_0, &oc);
    timer_channel_output_pulse_value_config(TIMER1, TIMER_CH_0, 3750U);  /* 50 % */
    timer_channel_output_mode_config(TIMER1, TIMER_CH_0, TIMER_OC_MODE_PWM0);
    timer_channel_output_shadow_config(TIMER1, TIMER_CH_0, TIMER_OC_SHADOW_DISABLE);
    timer_auto_reload_shadow_enable(TIMER1);
    timer_enable(TIMER1);
}

/* ============================================================================
 * SPI0 I²S master receive init
 * ============================================================================ */
static void _i2s_gpio_init(void)
{
    rcu_periph_clock_enable(RCU_GPIOA);   /* CK=PA5, SD=PA7 */
    rcu_periph_clock_enable(RCU_GPIOD);   /* L/R=PD5 */

    /* CK: PA5 — SPI0_SCK AF5 */
    gpio_af_set(CK_PORT, GPIO_AF_5, CK_PIN);
    gpio_mode_set(CK_PORT, GPIO_MODE_AF, GPIO_PUPD_NONE, CK_PIN);
    gpio_output_options_set(CK_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_100_220MHZ, CK_PIN);

    /* SD: PA7 — SPI0_MOSI AF5, input in I²S receive mode */
    gpio_af_set(SD_PORT, GPIO_AF_5, SD_PIN);
    gpio_mode_set(SD_PORT, GPIO_MODE_AF, GPIO_PUPD_PULLUP, SD_PIN);

    /* L/R: PD5 — GPIO output LOW = select left channel */
    gpio_mode_set(LR_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, LR_PIN);
    gpio_output_options_set(LR_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_60MHZ, LR_PIN);
    gpio_bit_reset(LR_PORT, LR_PIN);   /* LOW = left channel */
}

static void _i2s_periph_init(void)
{
    rcu_periph_clock_enable(RCU_SPI0);

    /* GD32H7xx SPL I²S API (confirmed from gd32h7xx_spi.h):
     *   i2s_init()     — sets mode, standard, clock polarity; deinits first
     *   i2s_psc_config() — sets sample rate, frame format, MCLK out
     *   i2s_enable()   — sets I2SEN bit
     *
     * Sample rate: 16 kHz, 16-bit data, 32-bit channel (Philips).
     * SPL i2s_psc_config() auto-calculates ODD+DIV from APB2 clock.         */
    i2s_init(SPI0, I2S_MODE_MASTERRX, I2S_STD_PHILIPS, I2S_CKPL_LOW);
    i2s_psc_config(SPI0, I2S_AUDIOSAMPLE_16K,
                   I2S_FRAMEFORMAT_DT16B_CH32B, I2S_MCKOUT_DISABLE);
    i2s_enable(SPI0);
}

/* ============================================================================
 * DMA0 CH0 — SPI0_RX → s_buf, circular, half+full ISR
 * ============================================================================ */
static void _dma_init(void)
{
    rcu_periph_clock_enable(RCU_DMA0);
    rcu_periph_clock_enable(RCU_DMAMUX);

    dma_deinit(DMA0, DMA_CH0);

    dma_single_data_parameter_struct p;
    dma_single_data_para_struct_init(&p);
    p.periph_addr         = (uint32_t)(&SPI_RDATA(SPI0));
    p.memory0_addr        = (uint32_t)s_buf;
    p.direction           = DMA_PERIPH_TO_MEMORY;
    p.number              = INMP441_BUF_SAMPLES;
    p.periph_inc          = DMA_PERIPH_INCREASE_DISABLE;
    p.memory_inc          = DMA_MEMORY_INCREASE_ENABLE;
    p.periph_memory_width = DMA_PERIPH_WIDTH_16BIT;
    p.circular_mode       = DMA_CIRCULAR_MODE_ENABLE;
    p.priority            = DMA_PRIORITY_MEDIUM;
    dma_single_data_mode_init(DMA0, DMA_CH0, &p);

    /* Route SPI0_RX request through DMAMUX channel 0 (DMA0_CH0). */
    dmamux_request_id_config(DMAMUX_MUXCH0, DMA_REQUEST_SPI0_RX);

    /* Enable half-transfer + full-transfer interrupts for ping-pong tracking */
    dma_interrupt_enable(DMA0, DMA_CH0, DMA_CHXCTL_HTFIE | DMA_CHXCTL_FTFIE);
    nvic_irq_enable(DMA0_Channel0_IRQn, 6U, 0U);

    spi_dma_enable(SPI0, SPI_DMA_RECEIVE);
    dma_channel_enable(DMA0, DMA_CH0);
}

/* ============================================================================
 * Public API
 * ============================================================================ */

void inmp441_init(void)
{
    _i2s_gpio_init();
    _i2s_periph_init();
    _dma_init();
    _ws_timer_init();   /* start WS last — ensures I²S is ready for first frame */
}

int16_t inmp441_get_sample(void)
{
    /* Return the sample just before the current DMA write pointer. */
    uint32_t idx = (s_write_idx == 0U) ? (INMP441_BUF_SAMPLES - 1U) : (s_write_idx - 1U);
    return s_buf[idx];
}

uint16_t inmp441_rms_256(void)
{
    uint32_t sum = 0U;
    uint32_t base;
    uint16_t i;

    /* Snapshot 256 samples ending at write pointer — no lock needed (circular). */
    base = (s_write_idx >= 256U) ? (s_write_idx - 256U) : (INMP441_BUF_SAMPLES + s_write_idx - 256U);
    for (i = 0U; i < 256U; i++) {
        int32_t v = s_buf[(base + i) % INMP441_BUF_SAMPLES];
        sum += (uint32_t)(v * v);
    }
    /* Integer sqrt via Newton's method */
    uint32_t mean_sq = sum / 256U;
    uint32_t r = mean_sq;
    if (r == 0U) return 0U;
    for (uint8_t k = 0U; k < 16U; k++) {
        r = (r + mean_sq / r) >> 1;
    }
    return (uint16_t)((r > 32767U) ? 32767U : r);
}

/* ============================================================================
 * DMA0 Channel0 ISR — update write pointer on half / full transfer
 * ============================================================================ */
void DMA0_Channel0_IRQHandler(void)
{
    if (dma_interrupt_flag_get(DMA0, DMA_CH0, DMA_INT_FLAG_HTF)) {
        dma_interrupt_flag_clear(DMA0, DMA_CH0, DMA_INT_FLAG_HTF);
        s_write_idx = INMP441_BUF_SAMPLES / 2U;
    }
    if (dma_interrupt_flag_get(DMA0, DMA_CH0, DMA_INT_FLAG_FTF)) {
        dma_interrupt_flag_clear(DMA0, DMA_CH0, DMA_INT_FLAG_FTF);
        s_write_idx = 0U;
    }
}

/****************************End*****************************/

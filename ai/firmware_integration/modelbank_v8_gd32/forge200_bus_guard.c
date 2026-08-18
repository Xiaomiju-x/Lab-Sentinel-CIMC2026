#include "forge200_bus_guard.h"

#include "HeaderFiles.h"
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"
#include "forge200_shared_spi.h"

#include <string.h>

#define F2_SD_CS_PORT GPIOC
#define F2_SD_CS_PIN GPIO_PIN_5
#define F2_MAX_CS_PORT GPIOG
#define F2_MAX_CS_PIN GPIO_PIN_3
#define F2_CLK_PORT GPIOB
#define F2_CLK_PIN GPIO_PIN_10
#define F2_MOSI_PORT GPIOC
#define F2_MOSI_PIN GPIO_PIN_1

static SemaphoreHandle_t s_mutex;
static SemaphoreHandle_t s_inference_mutex;
static forge200_shared_spi_t s_bus;
static forge200_bus_metrics_t s_metrics;
static TickType_t s_hold_started;
static uint8_t s_initialized;

static void hook_critical_enter(void *context)
{
    (void)context;
    taskENTER_CRITICAL();
}

static void hook_critical_exit(void *context)
{
    (void)context;
    taskEXIT_CRITICAL();
}

static void hook_sd_high(void *context)
{
    (void)context;
    gpio_bit_set(F2_SD_CS_PORT, F2_SD_CS_PIN);
}

static void hook_max_high(void *context)
{
    (void)context;
    gpio_bit_set(F2_MAX_CS_PORT, F2_MAX_CS_PIN);
}

static void hook_set_mode(void *context, forge200_spi_mode_t mode)
{
    (void)context;
    gpio_bit_reset(F2_CLK_PORT, F2_CLK_PIN);
    if (mode == FORGE200_SPI_MODE_SD_MODE0) {
        gpio_bit_set(F2_MOSI_PORT, F2_MOSI_PIN);
    } else {
        gpio_bit_reset(F2_MOSI_PORT, F2_MOSI_PIN);
    }
}

int forge200_bus_guard_init(void)
{
    forge200_spi_hooks_t hooks;
    if (s_initialized != 0U) {
        return 0;
    }
    s_mutex = xSemaphoreCreateMutex();
    if (s_mutex == NULL) {
        return -1;
    }
    s_inference_mutex = xSemaphoreCreateMutex();
    if (s_inference_mutex == NULL) {
        vSemaphoreDelete(s_mutex);
        s_mutex = NULL;
        return -2;
    }
    memset(&hooks, 0, sizeof(hooks));
    hooks.critical_enter = hook_critical_enter;
    hooks.critical_exit = hook_critical_exit;
    hooks.deassert_sd_cs_pc5 = hook_sd_high;
    hooks.deassert_max31856_cs_pg3 = hook_max_high;
    hooks.set_mode = hook_set_mode;
    if (forge200_shared_spi_init(&s_bus, &hooks) != 0) {
        vSemaphoreDelete(s_mutex);
        s_mutex = NULL;
        vSemaphoreDelete(s_inference_mutex);
        s_inference_mutex = NULL;
        return -3;
    }
    memset(&s_metrics, 0, sizeof(s_metrics));
    s_initialized = 1U;
    return 0;
}

int forge200_bus_guard_acquire(forge200_bus_owner_t owner, uint32_t timeout_ms)
{
    TickType_t begin;
    TickType_t acquired;
    TickType_t waited;
    forge200_spi_owner_t spi_owner;
    if (s_initialized == 0U || s_mutex == NULL ||
        (owner != FORGE200_BUS_OWNER_SD &&
         owner != FORGE200_BUS_OWNER_MAX31856)) {
        return -1;
    }
    begin = xTaskGetTickCount();
    if (xSemaphoreTake(s_mutex, pdMS_TO_TICKS(timeout_ms)) != pdTRUE) {
        taskENTER_CRITICAL();
        s_metrics.timeout_refusals++;
        taskEXIT_CRITICAL();
        return -2;
    }
    acquired = xTaskGetTickCount();
    waited = acquired - begin;
    spi_owner = owner == FORGE200_BUS_OWNER_SD
                    ? FORGE200_SPI_OWNER_SD
                    : FORGE200_SPI_OWNER_MAX31856;
    if (forge200_shared_spi_try_acquire(&s_bus, spi_owner) != 0) {
        taskENTER_CRITICAL();
        s_metrics.collision_refusals++;
        taskEXIT_CRITICAL();
        (void)xSemaphoreGive(s_mutex);
        return -3;
    }
    taskENTER_CRITICAL();
    if ((uint32_t)waited > s_metrics.max_wait_ticks) {
        s_metrics.max_wait_ticks = (uint32_t)waited;
    }
    if (owner == FORGE200_BUS_OWNER_SD) {
        s_metrics.sd_acquisitions++;
    } else {
        s_metrics.max31856_acquisitions++;
    }
    s_metrics.current_owner = (uint32_t)owner;
    s_hold_started = acquired;
    taskEXIT_CRITICAL();
    return 0;
}

int forge200_bus_guard_release(forge200_bus_owner_t owner)
{
    TickType_t held;
    forge200_spi_owner_t spi_owner;
    int result;
    if (s_initialized == 0U || s_mutex == NULL) {
        return -1;
    }
    spi_owner = owner == FORGE200_BUS_OWNER_SD
                    ? FORGE200_SPI_OWNER_SD
                    : FORGE200_SPI_OWNER_MAX31856;
    held = xTaskGetTickCount() - s_hold_started;
    result = forge200_shared_spi_release(&s_bus, spi_owner);
    taskENTER_CRITICAL();
    if ((uint32_t)held > s_metrics.max_hold_ticks) {
        s_metrics.max_hold_ticks = (uint32_t)held;
    }
    s_metrics.current_owner = 0U;
    taskEXIT_CRITICAL();
    if (xSemaphoreGive(s_mutex) != pdTRUE) {
        return -2;
    }
    return result;
}

void forge200_bus_guard_snapshot(forge200_bus_metrics_t *metrics)
{
    if (metrics == NULL) {
        return;
    }
    taskENTER_CRITICAL();
    *metrics = s_metrics;
    metrics->collision_refusals += s_bus.collision_refusals;
    taskEXIT_CRITICAL();
}

int forge200_inference_guard_acquire(uint32_t timeout_ms)
{
    if (s_initialized == 0U || s_inference_mutex == NULL) {
        return -1;
    }
    return xSemaphoreTake(
               s_inference_mutex, pdMS_TO_TICKS(timeout_ms)) == pdTRUE
               ? 0
               : -2;
}

int forge200_inference_guard_release(void)
{
    if (s_initialized == 0U || s_inference_mutex == NULL) {
        return -1;
    }
    return xSemaphoreGive(s_inference_mutex) == pdTRUE ? 0 : -2;
}

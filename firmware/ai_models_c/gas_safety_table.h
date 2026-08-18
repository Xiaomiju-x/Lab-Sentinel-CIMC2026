/* gas_safety_table.h  --  AUTO-GENERATED, do not edit by hand. */
/* Source: predict_engine/raw_materials.json (safety + notes fields). */
/* Generator: CIMC/model/gas_safety/gen_gas_safety_table.py            */
/* Formula-aware furnace gas-evolution safety table for the GD32 edge  */
/* sentinel: predicts hazardous gas vs furnace temperature, then       */
/* cross-checks the MQ-135 sensor. ASCII-only (armcc / no %f safe).     */
/* 14 rules over 13 hazardous materials.                 */
#ifndef GAS_SAFETY_TABLE_H
#define GAS_SAFETY_TABLE_H
#include <stdint.h>

/* hazardous gas species tracked */
enum { GAS_NONE=0, GAS_NH3, GAS_HF, GAS_NOX, GAS_CO2, GAS_SUBLIME, GAS_CR6, GAS_NSPECIES };
/* severity: 1=process note, 2=caution(ventilate), 3=danger(toxic/corrosive) */

typedef struct {
    const char *mat;     /* raw material formula (ASCII) */
    uint8_t     gas;     /* GAS_* species evolved on heating */
    int16_t     onset_c; /* temperature (C) where evolution is significant */
    uint8_t     sev;     /* severity 1..3 */
    const char *reason;  /* short ASCII rationale (from json safety/notes) */
} gas_rule_t;

static const char *const GAS_NAME[GAS_NSPECIES] =
    { "none", "NH3", "HF", "NOx", "CO2", "vapor", "Cr6+" };

static const gas_rule_t GAS_RULES[] = {
    { "NH4F", GAS_NH3, 200, 2, "ammonium fluoride decomposes >200C, co-evolves NH3" },
    { "NH4F", GAS_HF, 200, 3, "decomposes >200C to HF, corrosive, MUST ventilate" },
    { "(NH4)2HPO4", GAS_NH3, 200, 2, "decomposes ~200C releasing NH3, slow ramp" },
    { "NH4H2PO4", GAS_NH3, 190, 2, "DAP decomposes ~190C releasing NH3" },
    { "Cr(NO3)3.9H2O", GAS_NOX, 200, 3, "chromium nitrate decomposes ~200C, brown NOx, oxidizer" },
    { "Cu(NO3)2.3H2O", GAS_NOX, 170, 2, "copper nitrate decomposes ~170C releasing NOx" },
    { "Li2CO3", GAS_CO2, 700, 1, "Li carbonate calcines ~700C, CO2" },
    { "CaCO3", GAS_CO2, 700, 1, "CaCO3 calcines 700-825C, CO2 evolution" },
    { "Na2CO3", GAS_CO2, 850, 1, "Na2CO3 melts/decomp ~851C, CO2" },
    { "K2CO3", GAS_CO2, 891, 1, "K2CO3 melts/decomp ~891C, CO2" },
    { "SrCO3", GAS_CO2, 1100, 1, "SrCO3 decomposes ~1100C, CO2" },
    { "BaCO3", GAS_CO2, 1100, 2, "BaCO3 decomposes ~1100C, CO2 + Ba toxicity" },
    { "MoO3", GAS_SUBLIME, 700, 2, "MoO3 sublimes severely >700C, control temp/seal" },
    { "Cr2O3", GAS_CR6, 900, 2, "Cr6+ aerosol possible in oxidizing atmosphere at high T" },
};
#define GAS_RULES_N (sizeof(GAS_RULES)/sizeof(GAS_RULES[0]))

#endif /* GAS_SAFETY_TABLE_H */

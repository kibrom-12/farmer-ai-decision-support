from fertilizer_decision_engine import recommend_fertilizer

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


def _find_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def _num(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def diagnose_soil(soil_pH, soil_N_pct, soil_P_ppm):
    diagnosis = []

    if soil_pH < 5.5:
        diagnosis.append("Soil pH is strongly acidic.")
    elif soil_pH < 6.0:
        diagnosis.append("Soil pH is moderately acidic.")
    elif soil_pH <= 7.2:
        diagnosis.append("Soil pH is in a generally favorable range.")
    else:
        diagnosis.append("Soil pH is alkaline.")

    if soil_N_pct < 0.10:
        diagnosis.append("Total nitrogen appears low.")
    elif soil_N_pct < 0.20:
        diagnosis.append("Total nitrogen is in a moderate range.")
    else:
        diagnosis.append("Total nitrogen is relatively high.")

    if soil_P_ppm < 5:
        diagnosis.append("Available phosphorus appears very low.")
    elif soil_P_ppm < 10:
        diagnosis.append("Available phosphorus appears low.")
    elif soil_P_ppm < 20:
        diagnosis.append("Available phosphorus is in a moderate range.")
    else:
        diagnosis.append("Available phosphorus is relatively high.")

    return diagnosis


def _get_strategy_columns(df, strategy):
    if strategy == "NE":
        names = ["NE"]
    elif strategy == "Regional":
        names = ["Regional"]
    else:
        names = ["SoilTest", "Soil-Test", "Soil Test"]

    result = {
        "yield": None,
        "N": None,
        "P": None,
    }

    for prefix in names:
        result["yield"] = result["yield"] or _find_column(
            df,
            [
                f"{prefix}_yield_kg_ha",
                f"{prefix}_yield",
                f"{prefix} Yield",
                f"{prefix} Yield (kg/ha)",
            ],
        )

        result["N"] = result["N"] or _find_column(
            df,
            [
                f"{prefix}_N_kg_ha",
                f"{prefix}_N",
                f"{prefix} N",
            ],
        )

        result["P"] = result["P"] or _find_column(
            df,
            [
                f"{prefix}_P_kg_ha",
                f"{prefix}_P",
                f"{prefix} P",
            ],
        )

    return result


def recommend_fertilizer(
    soil_pH,
    soil_N_pct,
    soil_P_ppm,
    modeling_table,
):
    """
    Ethiopian fertilizer strategy recommendation using
    TAMASA fertilizer-response trial data.

    Soil K is intentionally omitted because the available
    TAMASA soil dataset does not contain complete soil-K
    measurements.

    Inputs:
        soil_pH     : soil pH
        soil_N_pct  : total soil nitrogen (%)
        soil_P_ppm  : available phosphorus (ppm)
        modeling_table : TAMASA modeling dataframe
    """

    df = modeling_table.copy()

    soil_pH = _num(soil_pH)
    soil_N_pct = _num(soil_N_pct)
    soil_P_ppm = _num(soil_P_ppm)

    diagnosis = diagnose_soil(
        soil_pH,
        soil_N_pct,
        soil_P_ppm,
    )

    # ---------------------------------------------------------
    # Find soil columns
    # ---------------------------------------------------------

    ph_col = _find_column(
        df,
        [
            "soil_pH",
            "soil_ph",
            "pH",
            "Soil pH",
           

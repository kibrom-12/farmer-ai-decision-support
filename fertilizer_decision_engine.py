"""
Fertilizer Decision Engine
--------------------------
Data-driven fertilizer strategy recommendation using Ethiopian TAMASA
performance-trial data.

Inputs:
    - Soil pH
    - Total soil nitrogen (%)
    - Available soil phosphorus (ppm)

K is intentionally omitted because the available TAMASA soil dataset does
not contain complete soil-K measurements.

This engine ranks the fertilizer strategies observed in the TAMASA trials:
    1. Control
    2. NE Recommendation
    3. Regional Recommendation
    4. Soil-Test Recommendation

It is a decision-support prototype, not a validated exact fertilizer-dose
calculator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


STRATEGIES = [
    "Control",
    "NE Recommendation",
    "Regional Recommendation",
    "Soil-Test Recommendation",
]

SOIL_FEATURES = ["soil_pH", "soil_N_pct", "soil_P_ppm"]

YIELD_COLUMNS = {
    "Control": "control_yield_kg_ha",
    "NE Recommendation": "ne_yield_kg_ha",
    "Regional Recommendation": "regional_yield_kg_ha",
    "Soil-Test Recommendation": "soiltest_yield_kg_ha",
}


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _prepare_modeling_table(modeling_table: pd.DataFrame) -> pd.DataFrame:
    df = modeling_table.copy()
    rename_map = {}

    pH_col = _find_column(df, ["soil_pH", "soil_ph", "pH", "ph", "Soil pH"])
    n_col = _find_column(
        df,
        ["soil_N_pct", "soil_N", "total_N_pct", "Total N (%)", "N_pct"],
    )
    p_col = _find_column(
        df,
        ["soil_P_ppm", "soil_P", "available_P_ppm", "Available P (ppm)", "P_ppm"],
    )

    if pH_col and pH_col != "soil_pH":
        rename_map[pH_col] = "soil_pH"
    if n_col and n_col != "soil_N_pct":
        rename_map[n_col] = "soil_N_pct"
    if p_col and p_col != "soil_P_ppm":
        rename_map[p_col] = "soil_P_ppm"

    yield_candidates = {
        "Control": ["control_yield_kg_ha", "Control Yield (kg/ha)", "control_yield"],
        "NE Recommendation": ["ne_yield_kg_ha", "NE Yield (kg/ha)", "NE_yield"],
        "Regional Recommendation": [
            "regional_yield_kg_ha",
            "Regional Yield (kg/ha)",
            "regional_yield",
        ],
        "Soil-Test Recommendation": [
            "soiltest_yield_kg_ha",
            "Soil-Test Yield (kg/ha)",
            "soil_test_yield",
        ],
    }

    for strategy, canonical in YIELD_COLUMNS.items():
        if canonical not in df.columns:
            col = _find_column(df, yield_candidates[strategy])
            if col:
                rename_map[col] = canonical

    if rename_map:
        df = df.rename(columns=rename_map)

    required = SOIL_FEATURES + list(YIELD_COLUMNS.values())
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "The TAMASA CSV is missing required columns: " + ", ".join(missing)
        )

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def diagnose_soil(
    soil_pH: float,
    soil_N_pct: float,
    soil_P_ppm: float,
) -> List[str]:
    diagnosis = []

    if soil_pH < 5.5:
        diagnosis.append(
            "Soil pH is strongly acidic; acidity management should be considered."
        )
    elif soil_pH < 6.0:
        diagnosis.append(
            "Soil pH is moderately acidic; pH management may improve nutrient availability."
        )
    elif soil_pH <= 7.2:
        diagnosis.append(
            "Soil pH is in a broadly suitable range for many crops."
        )
    else:
        diagnosis.append(
            "Soil pH is relatively high; phosphorus and micronutrient availability may need attention."
        )

    if soil_N_pct < 0.10:
        diagnosis.append(
            "Total soil nitrogen is low relative to the TAMASA observations."
        )
    elif soil_N_pct < 0.20:
        diagnosis.append(
            "Total soil nitrogen is in a moderate range relative to the TAMASA observations."
        )
    else:
        diagnosis.append(
            "Total soil nitrogen is relatively high compared with many TAMASA observations."
        )

    if soil_P_ppm < 10:
        diagnosis.append(
            "Available phosphorus is relatively low; phosphorus management is important."
        )
    elif soil_P_ppm < 20:
        diagnosis.append(
            "Available phosphorus is in an intermediate range."
        )
    else:
        diagnosis.append(
            "Available phosphorus is relatively high compared with many TAMASA observations."
        )

    diagnosis.append(
        "Soil potassium is not modeled because complete soil-K measurements "
        "were not available in the TAMASA soil data."
    )

    return diagnosis


def _predict_strategy_yield(
    strategy: str,
    df: pd.DataFrame,
    x_input: np.ndarray,
) -> Tuple[float, str]:
    y_col = YIELD_COLUMNS[strategy]
    subset = df[SOIL_FEATURES + [y_col]].dropna().copy()

    if subset.empty:
        return float("nan"), "No complete TAMASA observations"

    X = subset[SOIL_FEATURES].to_numpy(dtype=float)
    y = subset[y_col].to_numpy(dtype=float)

    if len(subset) < 3:
        return float(np.mean(y)), "Historical mean; too few observations"

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    input_scaled = scaler.transform(x_input)

    n_neighbors = min(7, len(subset))
    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights="distance",
    )
    model.fit(X_scaled, y)

    local_prediction = float(model.predict(input_scaled)[0])
    historical_mean = float(np.mean(y))

    prediction = 0.70 * local_prediction + 0.30 * historical_mean
    prediction = max(0.0, prediction)

    return prediction, "KNN soil-similarity estimate blended with historical mean"


def recommend_fertilizer(
    soil_pH: float,
    soil_N_pct: float,
    soil_P_ppm: float,
    modeling_table: pd.DataFrame,
) -> Dict:
    values = [soil_pH, soil_N_pct, soil_P_ppm]

    if any(pd.isna(v) for v in values):
        raise ValueError("Soil pH, soil N, and soil P must all be provided.")

    try:
        soil_pH = float(soil_pH)
        soil_N_pct = float(soil_N_pct)
        soil_P_ppm = float(soil_P_ppm)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Soil pH, soil N, and soil P must be numeric."
        ) from exc

    df = _prepare_modeling_table(modeling_table)

    x_input = np.array(
        [[soil_pH, soil_N_pct, soil_P_ppm]],
        dtype=float,
    )

    estimated_yields: Dict[str, float] = {}
    methods: Dict[str, str] = {}

    for strategy in STRATEGIES:
        estimate, method = _predict_strategy_yield(
            strategy,
            df,
            x_input,
        )
        estimated_yields[strategy] = estimate
        methods[strategy] = method

    valid = {
        strategy: value
        for strategy, value in estimated_yields.items()
        if np.isfinite(value)
    }

    if not valid:
        raise ValueError(
            "No usable yield observations were found in the TAMASA CSV."
        )

    ranked = sorted(
        valid.keys(),
        key=lambda name: valid[name],
        reverse=True,
    )

    recommended = ranked[0]
    recommended_yield = float(valid[recommended])

    if len(ranked) >= 2:
        second_yield = float(valid[ranked[1]])
        margin = recommended_yield - second_yield
    else:
        margin = float("nan")

    if np.isfinite(margin) and margin >= 500:
        confidence = "Moderate"
    else:
        confidence = "Low"

    diagnosis = diagnose_soil(
        soil_pH=soil_pH,
        soil_N_pct=soil_N_pct,
        soil_P_ppm=soil_P_ppm,
    )

    reason = (
        f"The model estimates the highest yield among the available TAMASA "
        f"fertilizer strategies for the entered soil conditions under "
        f"'{recommended}'. The estimated yield is "
        f"{recommended_yield:,.0f} kg/ha."
    )

    warning = (
        "This is a fertilizer-strategy decision-support prototype based on "
        "Ethiopian TAMASA performance trials. It is not a validated exact "
        "fertilizer-dose calculator. K is omitted because complete soil-K "
        "measurements were unavailable. Exact N/P/K rates should only be "
        "shown after additional fertilizer-response data and local validation."
    )

    return {
        "recommended_strategy": recommended,
        "confidence": confidence,
        "estimated_yield_kg_ha": recommended_yield,
        "margin_kg_ha": margin,
        "ranked_strategies": ranked,
        "estimated_yield_by_strategy": estimated_yields,
        "soil_diagnosis": diagnosis,
        "reason": reason,
        "warning": warning,
        "prediction_methods": methods,
    }


def load_tamasa_table(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"TAMASA fertilizer data file not found: {csv_path}"
        )

    return pd.read_csv(csv_path)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    csv_file = here / "TAMASA_fertilizer_modeling_table.csv"

    if csv_file.exists():
        table = load_tamasa_table(csv_file)

        result = recommend_fertilizer(
            soil_pH=6.0,
            soil_N_pct=0.15,
            soil_P_ppm=7.0,
            modeling_table=table,
        )

        print("Recommended strategy:", result["recommended_strategy"])
        print("Confidence:", result["confidence"])
        print(
            "Estimated yield:",
            f'{result["estimated_yield_kg_ha"]:,.0f} kg/ha',
        )
    else:
        print(
            "Place TAMASA_fertilizer_modeling_table.csv beside this file "
            "to run the local test."
        )

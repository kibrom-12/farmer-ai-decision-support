# fertilizer_decision_engine.py

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


# =========================================================
# 1. PARSE FERTILIZER N-P-K RATE
# =========================================================
def parse_npk_rate(value):
    """
    Convert fertilizer rate such as:
        120-50-31

    into:
        N = 120
        P = 50
        K = 31
    """

    if pd.isna(value):
        return np.nan, np.nan, np.nan

    text = str(value).strip()

    try:
        parts = text.split("-")

        if len(parts) >= 3:
            n = float(parts[0])
            p = float(parts[1])
            k = float(parts[2])

            return n, p, k

    except Exception:
        pass

    return np.nan, np.nan, np.nan


# =========================================================
# 2. SOIL DIAGNOSIS
# =========================================================
def diagnose_soil(
    soil_pH,
    soil_N_pct,
    soil_P_ppm
):
    """
    Basic soil diagnosis using the variables available
    in the TAMASA fertilizer-response dataset.
    """

    diagnosis = []

    # -----------------------------------------------------
    # pH
    # -----------------------------------------------------
    if soil_pH < 5.5:

        diagnosis.append(
            "Soil pH is strongly acidic."
        )

    elif soil_pH < 6.0:

        diagnosis.append(
            "Soil pH is moderately acidic."
        )

    elif soil_pH <= 7.5:

        diagnosis.append(
            "Soil pH is within a generally suitable range "
            "for many crops."
        )

    else:

        diagnosis.append(
            "Soil pH is alkaline."
        )

    # -----------------------------------------------------
    # Nitrogen
    # -----------------------------------------------------
    if soil_N_pct < 0.10:

        diagnosis.append(
            "Total soil nitrogen is low."
        )

    elif soil_N_pct < 0.20:

        diagnosis.append(
            "Total soil nitrogen is moderate."
        )

    else:

        diagnosis.append(
            "Total soil nitrogen is relatively high."
        )

    # -----------------------------------------------------
    # Phosphorus
    # -----------------------------------------------------
    if soil_P_ppm < 10:

        diagnosis.append(
            "Available phosphorus is low."
        )

    elif soil_P_ppm < 20:

        diagnosis.append(
            "Available phosphorus is moderate."
        )

    else:

        diagnosis.append(
            "Available phosphorus is relatively high."
        )

    return diagnosis


# =========================================================
# 3. FERTILIZER RECOMMENDATION ENGINE
# =========================================================
def recommend_fertilizer(
    soil_pH,
    soil_N_pct,
    soil_P_ppm,
    modeling_table
):
    """
    Evaluate Ethiopian fertilizer strategies using:

        Soil pH
        Total soil N
        Available soil P

    Strategies:

        1. NE Recommendation
        2. Regional Recommendation
        3. Soil Test Based Recommendation

    The model estimates expected yield for each strategy
    using historical Ethiopian TAMASA trials.

    IMPORTANT:
    This is a fertilizer STRATEGY recommendation.
    It is not a guaranteed exact fertilizer prescription.
    """

    # -----------------------------------------------------
    # Copy dataset
    # -----------------------------------------------------
    df = modeling_table.copy()

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------
    required_columns = [
        "soil_pH",
        "soil_total_N_pct",
        "soil_available_P_ppm",
        "NE_yield_kg_ha",
        "Regional_yield_kg_ha",
        "SoilTest_yield_kg_ha"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "The fertilizer dataset is missing required "
            f"columns: {missing_columns}"
        )

    # -----------------------------------------------------
    # Remove incomplete soil records
    # -----------------------------------------------------
    model_df = df.dropna(
        subset=[
            "soil_pH",
            "soil_total_N_pct",
            "soil_available_P_ppm"
        ]
    ).copy()

    if len(model_df) < 10:

        raise ValueError(
            "There are not enough complete Ethiopian "
            "soil trials for fertilizer recommendation."
        )

    # -----------------------------------------------------
    # Model features
    # -----------------------------------------------------
    feature_columns = [
        "soil_pH",
        "soil_total_N_pct",
        "soil_available_P_ppm"
    ]

    X = model_df[
        feature_columns
    ].astype(float)

    farmer_X = pd.DataFrame(
        [[
            float(soil_pH),
            float(soil_N_pct),
            float(soil_P_ppm)
        ]],
        columns=feature_columns
    )

    # -----------------------------------------------------
    # Fertilizer strategies
    # -----------------------------------------------------
    strategies = {
        "NE Recommendation": "NE_yield_kg_ha",
        "Regional Recommendation": "Regional_yield_kg_ha",
        "Soil Test Based Recommendation": "SoilTest_yield_kg_ha"
    }

    predictions = {}

    # =====================================================
    # 4. TRAIN ONE LOCAL KNN MODEL FOR EACH STRATEGY
    # =====================================================
    for strategy, yield_column in strategies.items():

        valid = model_df.dropna(
            subset=[
                "soil_pH",
                "soil_total_N_pct",
                "soil_available_P_ppm",
                yield_column
            ]
        ).copy()

        if len(valid) < 5:
            continue

        X_valid = valid[
            feature_columns
        ].astype(float)

        y_valid = valid[
            yield_column
        ].astype(float)

        # -------------------------------------------------
        # KNN
        # -------------------------------------------------
        n_neighbors = min(
            7,
            len(valid)
        )

        model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights="distance"
        )

        model.fit(
            X_valid,
            y_valid
        )

        local_prediction = model.predict(
            farmer_X
        )[0]

        # -------------------------------------------------
        # Historical mean
        # -------------------------------------------------
        historical_mean = y_valid.mean()

        # -------------------------------------------------
        # Stabilized prediction
        #
        # 70% local soil similarity
        # 30% historical Ethiopian performance
        # -------------------------------------------------
        final_prediction = (
            0.70 * local_prediction
            +
            0.30 * historical_mean
        )

        predictions[strategy] = float(
            final_prediction
        )

    # -----------------------------------------------------
    # Check prediction result
    # -----------------------------------------------------
    if not predictions:

        raise ValueError(
            "No fertilizer strategy could be evaluated "
            "from the available dataset."
        )

    # =====================================================
    # 5. RANK FERTILIZER STRATEGIES
    # =====================================================
    ranked_strategies = sorted(
        predictions.keys(),
        key=predictions.get,
        reverse=True
    )

    recommended_strategy = ranked_strategies[0]

    best_yield = predictions[
        recommended_strategy
    ]

    # -----------------------------------------------------
    # Second-best strategy
    # -----------------------------------------------------
    if len(ranked_strategies) > 1:

        second_best = ranked_strategies[1]

        second_yield = predictions[
            second_best
        ]

    else:

        second_yield = best_yield

    # -----------------------------------------------------
    # Difference between first and second

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor, CatBoostClassifier

APP_DIR = Path(__file__).parent

st.set_page_config(
    page_title="AI Farmer Decision Support",
    page_icon="🌾",
    layout="wide"
)

# ============================================================
# CROP LIST
# ============================================================

CROP_NAMES = {
    "1.0": "Barley",
    "2.0": "Maize",
    "6.0": "Sorghum",
    "8.0": "Wheat",
    "10.0": "Cassava",
    "12.0": "Haricot Beans",
    "13.0": "Horse Beans",
    "19.0": "Red Kidney Beans",
    "24.0": "Ground Nuts",
    "26.0": "Rape Seed",
    "28.0": "Sunflower",
    "38.0": "Red Pepper",
    "42.0": "Bananas",
    "46.0": "Mangos",
    "47.0": "Oranges",
    "48.0": "Papaya",
    "55.0": "Garlic",
    "56.0": "Kale",
    "61.0": "Pumpkins",
    "62.0": "Sweet Potato",
    "71.0": "Chat",
    "72.0": "Coffee",
    "74.0": "Enset",
    "75.0": "Gesho",
    "76.0": "Sugar Cane",
    "84.0": "Avocados",
    "98.0": "Other Root Crops"
}

CROP_GROUPS = {
    "1.0": "Cereal",
    "2.0": "Cereal",
    "6.0": "Cereal",
    "8.0": "Cereal",
    "10.0": "Root/Tuber",
    "62.0": "Root/Tuber",
    "98.0": "Root/Tuber",
    "12.0": "Legume",
    "13.0": "Legume",
    "19.0": "Legume",
    "24.0": "Legume",
    "26.0": "Oilseed",
    "28.0": "Oilseed",
    "38.0": "Vegetable",
    "55.0": "Vegetable",
    "56.0": "Vegetable",
    "61.0": "Vegetable",
    "42.0": "Fruit",
    "46.0": "Fruit",
    "47.0": "Fruit",
    "48.0": "Fruit",
    "84.0": "Fruit",
    "71.0": "Specialty",
    "72.0": "Perennial",
    "74.0": "Perennial",
    "75.0": "Perennial",
    "76.0": "Perennial"
}

CROP_CODES = list(CROP_NAMES.keys())

# ============================================================
# YIELD FEATURES
# ============================================================

REC_FEATURES = [
    "saq14", "saq01", "saq02", "saq03", "saq04",
    "saq05", "saq06", "saq07", "saq15",
    "s4q01b",
    "s3q02a", "s3q02b", "s3q03", "s3q04",
    "s3q05", "s3q07", "s3q08",
    "s3q12", "s3q16", "s3q28", "s3q35",
    "s3q36", "s3q38", "s3q40", "s3q42",
    "dist_road", "dist_market", "dist_popcenter",
    "ssa_aez09", "twi",
    "sq1", "sq2", "sq3", "sq4", "sq5", "sq6", "sq7",
    "af_bio_1", "af_bio_8", "af_bio_12",
    "af_bio_13", "af_bio_16",
    "slopepct", "srtm1k", "popdensity",
    "cropshare",
    "anntot_avg", "wetQ_avgstart", "wetQ_avg",
    "ndvi_avg", "lat_mod", "lon_mod"
]

CATEGORICAL_FEATURES = [
    "saq14", "saq01", "saq02", "saq06",
    "saq07", "saq15", "s4q01b",
    "s3q02b", "s3q03", "s3q04", "s3q05",
    "s3q07", "s3q12", "s3q16", "s3q35",
    "s3q36", "s3q38", "s3q40", "s3q42",
    "ssa_aez09",
    "sq1", "sq2", "sq3", "sq4",
    "sq5", "sq6", "sq7"
]

# ============================================================
# RAINFALL FEATURES
# ============================================================

RAIN_FEATURES = [
    "rain_lag_1",
    "rain_lag_2",
    "rain_lag_3",
    "rain_lag_7",
    "rain_lag_14",
    "rain_lag_21",
    "rain_lag_28",
    "rain_roll_3",
    "rain_roll_7",
    "rain_roll_14",
    "rain_roll_28",
    "temp_lag_1",
    "temp_lag_7",
    "temp_lag_14",
    "day_of_year",
    "month",
    "sin_doy",
    "cos_doy"
]

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    yield_model = CatBoostRegressor()
    yield_model.load_model(
        str(APP_DIR / "yield_model.cbm")
    )

    rain_event_model = CatBoostClassifier()
    rain_event_model.load_model(
        str(APP_DIR / "rain_event_model.cbm")
    )

    rain_amount_model = CatBoostRegressor()
    rain_amount_model.load_model(
        str(APP_DIR / "rain_amount_model.cbm")
    )

    return (
        yield_model,
        rain_event_model,
        rain_amount_model
    )


@st.cache_data
def load_rainfall():

    return pd.read_csv(
        APP_DIR / "rainfall_history.csv"
    )


# ============================================================
# BUILD FARM RECORD
# ============================================================

def build_farm(area_ha, latitude, longitude, crop_code):

    row = {}

    for feature in REC_FEATURES:
        row[feature] = 0

    row["s4q01b"] = str(crop_code)

    row["s3q08"] = float(area_ha) * 10000

    row["lat_mod"] = float(latitude)

    row["lon_mod"] = float(longitude)

    for feature in CATEGORICAL_FEATURES:

        if feature == "s4q01b":
            row[feature] = str(crop_code)
        else:
            row[feature] = "MISSING"

    return pd.DataFrame(
        [row],
        columns=REC_FEATURES
    )


# ============================================================
# CROP PREDICTION + DECISION DEGREE
# ============================================================

def predict_crops(area_ha, latitude, longitude):

    yield_model, _, _ = load_models()

    results = []

    for code in CROP_CODES:

        farm = build_farm(
            area_ha,
            latitude,
            longitude,
            code
        )

        predicted_log_yield = float(
            yield_model.predict(farm)[0]
        )

        predicted_yield = max(
            0.0,
            float(np.expm1(predicted_log_yield))
        )

        results.append({
            "Crop": CROP_NAMES[code],
            "Crop Code": code,
            "Crop Group": CROP_GROUPS[code],
            "Predicted Yield (kg/ha)": predicted_yield
        })

    result = pd.DataFrame(results)

    minimum = result[
        "Predicted Yield (kg/ha)"
    ].min()

    maximum = result[
        "Predicted Yield (kg/ha)"
    ].max()

    # Decision degree is normalized yield performance
    # from 0 to 100 across the evaluated crops.
    if maximum > minimum:

        result["Decision Degree (%)"] = (
            (

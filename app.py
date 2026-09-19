import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor, CatBoostClassifier
import plotly.express as px

# ============================================================
# AI FARMER DECISION SUPPORT SYSTEM
# ============================================================

st.set_page_config(
    page_title="AI Farmer Decision Support System",
    page_icon="🌾",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent

# ============================================================
# CROP CONFIGURATION
# ============================================================

CROP_NAMES = {
    "1.0": "Barley",
    "2.0": "Maize",
    "3.0": "Teff",
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
    "98.0": "Other Root Crops",
}

CROP_GROUPS = {
    "Barley": "Cereal",
    "Maize": "Cereal",
    "Teff": "Cereal",
    "Sorghum": "Cereal",
    "Wheat": "Cereal",
    "Cassava": "Root/Tuber",
    "Sweet Potato": "Root/Tuber",
    "Other Root Crops": "Root/Tuber",
    "Haricot Beans": "Legume",
    "Horse Beans": "Legume",
    "Red Kidney Beans": "Legume",
    "Ground Nuts": "Legume",
    "Rape Seed": "Oilseed",
    "Sunflower": "Oilseed",
    "Red Pepper": "Vegetable",
    "Garlic": "Vegetable",
    "Kale": "Vegetable",
    "Pumpkins": "Vegetable",
    "Bananas": "Fruit",
    "Mangos": "Fruit",
    "Oranges": "Fruit",
    "Papaya": "Fruit",
    "Avocados": "Fruit",
    "Chat": "Specialty",
    "Coffee": "Perennial",
    "Enset": "Perennial",
    "Gesho": "Perennial",
    "Sugar Cane": "Perennial",
}

# ============================================================
# YIELD MODEL FEATURES
# ============================================================

REC_FEATURES = [
    "saq14",
    "saq01",
    "saq02",
    "saq03",
    "saq04",
    "saq05",
    "saq06",
    "saq07",
    "saq15",
    "s4q01b",
    "s3q02a",
    "s3q02b",
    "s3q03",
    "s3q04",
    "s3q05",
    "s3q07",
    "s3q08",
    "s3q12",
    "s3q16",
    "s3q28",
    "s3q35",
    "s3q36",
    "s3q38",
    "s3q40",
    "s3q42",
    "dist_road",
    "dist_market",
    "dist_popcenter",
    "ssa_aez09",
    "twi",
    "sq1",
    "sq2",
    "sq3",
    "sq4",
    "sq5",
    "sq6",
    "sq7",
    "af_bio_1",
    "af_bio_8",
    "af_bio_12",
    "af_bio_13",
    "af_bio_16",
    "slopepct",
    "srtm1k",
    "popdensity",
    "cropshare",
    "anntot_avg",
    "wetQ_avgstart",
    "wetQ_avg",
    "ndvi_avg",
    "lat_mod",
    "lon_mod",
]

CATEGORICAL_FEATURES = [
    "saq14",
    "saq01",
    "saq02",
    "saq06",
    "saq07",
    "saq15",
    "s4q01b",
    "s3q02b",
    "s3q03",
    "s3q04",
    "s3q05",
    "s3q07",
    "s3q12",
    "s3q16",
    "s3q35",
    "s3q36",
    "s3q38",
    "s3q40",
    "s3q42",
    "ssa_aez09",
    "sq1",
    "sq2",
    "sq3",
    "sq4",
    "sq5",
    "sq6",
    "sq7",
]

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
    "cos_doy",
]

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_yield_model():
    model = CatBoostRegressor()
    model.load_model(str(APP_DIR / "yield_model.cbm"))
    return model


@st.cache_resource
def load_rain_models():
    event_model = CatBoostClassifier()
    event_model.load_model(str(APP_DIR / "rain_event_model.cbm"))

    amount_model = CatBoostRegressor()
    amount_model.load_model(str(APP_DIR / "rain_amount_model.cbm"))

    return event_model, amount_model


@st.cache_data
def load_rainfall_history():
    return pd.read_csv(APP_DIR / "rainfall_history.csv")


@st.cache_data
def load_candidate_crops():
    path = APP_DIR / "candidate_crops.csv"

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame({
        "crop_code": list(CROP_NAMES.keys())
    })


@st.cache_data
def load_fertilizer_data():
    return pd.read_csv(
        APP_DIR / "TAMASA_fertilizer_modeling_table.csv"
    )


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

def check_required_files():

    required = [
        "yield_model.cbm",
        "rain_event_model.cbm",
        "rain_amount_model.cbm",
        "rainfall_history.csv",
        "candidate_crops.csv",
        "fertilizer_decision_engine.py",
        "TAMASA_fertilizer_modeling_table.csv",
    ]

    return {
        filename: (APP_DIR / filename).exists()
        for filename in required
    }


# ============================================================
# FARM PROFILE
# ============================================================

def build_farm(
    region,
    zone,
    woreda,
    area_ha,
    lat,
    lon,
    ph,
    irrigation,
    cropping_method,
    fallow,
    crop_share,
    slope,
    annual_rainfall,
    annual_temperature,
):

    farm = {
        feature: np.nan
        for feature in REC_FEATURES
    }

    farm.update({
        "saq14": "1",
        "saq01": region,
        "saq02": zone,
        "saq03": woreda,
        "saq04": cropping_method,
        "saq05": fallow,
        "saq06": irrigation,
        "saq07": "1",
        "saq15": "1",

        "s3q02a": lat,
        "s3q02b": lon,
        "s3q03": cropping

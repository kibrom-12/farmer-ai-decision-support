import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

from catboost import CatBoostRegressor, CatBoostClassifier

from fertilizer_decision_engine import recommend_fertilizer


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Farmer Decision Support",
    page_icon="🌾",
    layout="wide"
)

APP_DIR = Path(__file__).parent


# ============================================================
# CROP NAMES
# IMPORTANT:
# These are the crop codes supported by the finalized
# yield model.
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


# ============================================================
# CROP GROUPS
# ============================================================

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


# ============================================================
# CROP pH RANGES
# ============================================================

CROP_PH_RANGES = {
    "1.0": (6.0, 7.5),
    "2.0": (5.8, 7.0),
    "6.0": (5.5, 7.5),
    "8.0": (6.0, 7.0),

    "10.0": (5.5, 6.5),
    "12.0": (6.0, 7.0),
    "13.0": (6.0, 7.2),
    "19.0": (6.0, 7.0),
    "24.0": (5.3, 6.5),

    "26.0": (5.8, 7.5),
    "28.0": (6.0, 7.2),
    "38.0": (6.0, 7.0),

    "42.0": (5.5, 6.5),
    "46.0": (5.5, 7.5),
    "47.0": (5.5, 6.5),
    "48.0": (6.0, 6.5),

    "55.0": (6.0, 7.0),
    "56.0": (6.0, 7.5),
    "61.0": (6.0, 7.5),
    "62.0": (5.5, 6.8),

    "71.0": (6.0, 7.0),
    "72.0": (5.0, 6.0),
    "74.0": (5.5, 6.5),
    "75.0": (6.0, 7.0),
    "76.0": (6.0, 7.5),

    "84.0": (5.5, 6.5),
    "98.0": (5.5, 6.5)
}


CROP_CODES = list(CROP_NAMES.keys())


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
    "lon_mod"
]

    "dist_road",
    "dist_market",

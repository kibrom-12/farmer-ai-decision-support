import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


# ============================================================
# APP CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).parent

st.set_page_config(
    page_title="AI Farmer Decision Support",
    page_icon="🌾",
    layout="wide"
)


# ============================================================
# CROP DEFINITIONS
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

import os
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor, CatBoostClassifier

st.set_page_config(
    page_title="AI Farmer Decision Support",
    page_icon="🌱",
    layout="wide"
)

APP_DIR = os.path.dirname(__file__)

# =========================
# LOAD MODELS AND DATA
# =========================

yield_model = CatBoostRegressor()
yield_model.load_model(os.path.join(APP_DIR, "yield_model.cbm"))

rain_event_model = CatBoostClassifier()
rain_event_model.load_model(os.path.join(APP_DIR, "rain_event_model.cbm"))

rain_amount_model = CatBoostRegressor()
rain_amount_model.load_model(os.path.join(APP_DIR, "rain_amount_model.cbm"))

crops = pd.read_csv(os.path.join(APP_DIR, "candidate_crops.csv"))
rain_history = pd.read_csv(os.path.join(APP_DIR, "rainfall_history.csv"))
rain_history["time"] = pd.to_datetime(rain_history["time"])

official_names = {
    "1.0":"Barley","2.0":"Maize","6.0":"Sorghum","8.0":"Wheat",
    "10.0":"Cassava","12.0":"Haricot Beans","13.0":"Horse Beans",
    "19.0":"Red Kidney Beans","24.0":"Ground Nuts","26.0":"Rape Seed",
    "28.0":"Sunflower","38.0":"Red Pepper","42.0":"Bananas",
    "46.0":"Mangos","47.0":"Oranges","48.0":"Papaya","55.0":"Garlic",
    "56.0":"Kale","61.0":"Pumpkins","62.0":"Sweet Potato",
    "71.0":"Chat","72.0":"Coffee","74.0":"Enset","75.0":"Gesho",
    "76.0":"Sugar Cane","84.0":"Avocados","98.0":"Other Root Crops"
}

groups = {
    "1.0":"cereal","2.0":"cereal","6.0":"cereal","8.0":"cereal",
    "10.0":"root_tuber","62.0":"root_tuber","98.0":"root_tuber",
    "12.0":"legume","13.0":"legume","19.0":"legume","24.0":"legume",
    "26.0":"oilseed","28.0":"oilseed",
    "38.0":"vegetable","55.0":"vegetable","56.0":"vegetable","61.0":"vegetable",
    "42.0":"fruit","46.0":"fruit","47.0":"fruit","48.0":"fruit","84.0":"fruit",
    "71.0":"specialty","72.0":"perennial","74.0":"perennial",
    "75.0":"perennial","76.0":"perennial"
}

rec_features = [
    "saq14","saq01","saq02","saq03","saq04","saq05","saq06","saq07","saq15",
    "s4q01b","s3q02a","s3q02b","s3q03","s3q04","s3q05","s3q07","s3q08",
    "s3q12","s3q16","s3q28","s3q35","s3q36","s3q38","s3q40","s3q42",
    "dist_road","dist_market","dist_popcenter","ssa_aez09","twi",
    "sq1","sq2","sq3","sq4","sq5","sq6","sq7","af_bio_1","af_bio_8",
    "af_bio_12","af_bio_13","af_bio_16","slopepct","srtm1k","popdensity",
    "cropshare","anntot_avg","wetQ_avgstart","wetQ_avg","ndvi_avg",
    "lat_mod","lon_mod"
]

categorical = [
    "saq14","saq01","saq02","saq06","saq07","saq15","s4q01b",
    "s3q02b","s3q03","s3q04","s3q05","s3q07","s3q12","s3q16","s3q35",
    "s3q36","s3q38","s3q40","s3q42","ssa_aez09","sq1","sq2","sq3","sq4",
    "sq5","sq6","sq7"
]

rain_features = [
    "rain_lag_1","rain_lag_2","rain_lag_3","rain_lag_7",
    "rain_lag_14","rain_lag_21","rain_lag_28",
    "rain_roll_3","rain_roll_7","rain_roll_14","rain_roll_28",
    "temp_lag_1","temp_lag_7","temp_lag_14",
    "dayofyear","month","sin_doy","cos_doy"
]

# =========================
# FUNCTIONS
# =========================

def build_farm(region, zone, woreda, area_ha, lat, lon):
    x = {c: 0 for c in rec_features}
    x["saq14"] = region
    x["saq01"] = zone
    x["saq02"] = woreda
    x["s3q08"] = area_ha * 10000
    x["lat_mod"] = lat
    x["lon_mod"] = lon
    x["s4q01b"] = "1.0"
    return pd.DataFrame([x])

def predict_crops(region, zone, woreda, area_ha, lat, lon):
    base = build_farm(region, zone, woreda, area_ha, lat, lon)
    rows = []

    for code in crops["s4q01b"].astype(str):
        x = base.copy()
        x["s4q01b"] = code

        for c in categorical:
            x[c] = x[c].fillna("MISSING").astype(str)

        for c in rec_features:
            if c not in categorical:
                x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0)

        log_pred = float(yield_model.predict(x)[0])
        pred = max(0, np.expm1(log_pred))

        rows.append({
            "Crop": official_names.get(code, code),
            "Group": groups.get(code, "other"),
            "Predicted yield (kg/ha)": pred
        })

    r = pd.DataFrame(rows)

    lo = r["Predicted yield (kg/ha)"].min()
    hi = r["Predicted yield (kg/ha)"].max()

    if hi > lo:
        r["Yield score"] = (
            (r["Predicted yield (kg/ha)"] - lo) /
            (hi - lo) * 100
        )
    else:
        r["Yield score"] = 100

    # Historical reliability proxy used by the deployment prototype
    # repaired: removed malformed reliability reference

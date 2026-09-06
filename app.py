import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor, CatBoostClassifier

APP_DIR = Path(__file__).parent

st.set_page_config(page_title="AI Farmer Decision Support", page_icon="🌾", layout="wide")

@st.cache_resource
def load_models():
    yield_model = CatBoostRegressor()
    yield_model.load_model(str(APP_DIR / "yield_model.cbm"))
    rain_event_model = CatBoostClassifier()
    rain_event_model.load_model(str(APP_DIR / "rain_event_model.cbm"))
    rain_amount_model = CatBoostRegressor()
    rain_amount_model.load_model(str(APP_DIR / "rain_amount_model.cbm"))
    return yield_model, rain_event_model, rain_amount_model

@st.cache_data
def load_data():
    crops = pd.read_csv(APP_DIR / "candidate_crops.csv")
    rainfall = pd.read_csv(APP_DIR / "rainfall_history.csv")
    return crops, rainfall

CROP_NAMES = {
    "1.0":"Barley", "2.0":"Maize", "6.0":"Sorghum", "8.0":"Wheat",
    "10.0":"Cassava", "12.0":"Haricot Beans", "13.0":"Horse Beans",
    "19.0":"Red Kidney Beans", "24.0":"Ground Nuts", "26.0":"Rape Seed",
    "28.0":"Sunflower", "38.0":"Red Pepper", "42.0":"Bananas",
    "46.0":"Mangos", "47.0":"Oranges", "48.0":"Papaya", "55.0":"Garlic",
    "56.0":"Kale", "61.0":"Pumpkins", "62.0":"Sweet Potato", "71.0":"Chat",
    "72.0":"Coffee", "74.0":"Enset", "75.0":"Gesho", "76.0":"Sugar Cane",
    "84.0":"Avocados", "98.0":"Other Root Crops"
}

CROP_GROUPS = {
    "1.0":"cereal","2.0":"cereal","6.0":"cereal","8.0":"cereal",
    "10.0":"root_tuber","62.0":"root_tuber","98.0":"root_tuber",
    "12.0":"legume","13.0":"legume","19.0":"legume","24.0":"legume",
    "26.0":"oilseed","28.0":"oilseed",
    "38.0":"vegetable","55.0":"vegetable","56.0":"vegetable","61.0":"vegetable",
    "42.0":"fruit","46.0":"fruit","47.0":"fruit","48.0":"fruit","84.0":"fruit",
    "71.0":"specialty",
    "72.0":"perennial","74.0":"perennial","75.0":"perennial","76.0":"perennial"
}

REC_FEATURES = [
    "saq14","saq01","saq02","saq03","saq04","saq05","saq06","saq07","saq15",
    "s4q01b", "s3q02a","s3q02b","s3q03","s3q04","s3q05","s3q07","s3q08",
    "s3q12","s3q16","s3q28","s3q35","s3q36","s3q38","s3q40","s3q42",
    "dist_road","dist_market","dist_popcenter",
    "ssa_aez09","twi",
    "sq1","sq2","sq3","sq4","sq5","sq6","sq7",
    "af_bio_1","af_bio_8","af_bio_12","af_bio_13","af_bio_16",
    "slopepct","srtm1k","popdensity","cropshare",
    "anntot_avg","wetQ_avgstart","wetQ_avg","ndvi_avg",
    "lat_mod","lon_mod"
]

CATEGORICAL = [
    "saq14","saq01","saq02","saq06","saq07","saq15","s4q01b",
    "s3q02b","s3q03","s3q04","s3q05","s3q07","s3q12","s3q16","s3q35",
    "s3q36","s3q38","s3q40","s3q42","ssa_aez09","sq1","sq2","sq3",
    "sq4","sq5","sq6","sq7"
]

def fertilizer(ph):
    if ph < 5.5:
        return ("Phosphorus-containing fertilizer", "Strongly acidic soil: acidity management should be considered using soil testing/local extension advice.")
    if ph < 6.5:
        return ("Nitrogen + phosphorus fertilizer", "Acidic soil: nitrogen and phosphorus support may be appropriate.")
    if ph <= 7.0:
        return ("Balanced NPK/compound fertilizer", "Near-neutral soil: balanced nutrient management is appropriate.")
    if ph <= 7.5:
        return ("Balanced NPK/compound fertilizer", "Slightly alkaline soil: monitor nutrient availability.")
    return ("Phosphorus-containing or balanced fertilizer", "Alkaline soil can affect nutrient availability; use soil-test/local guidance.")

def build_farm(area_ha, lat, lon, crop_code):
    row = {c: 0 for c in REC_FEATURES}
    row["s4q01b"] = str(crop_code)
    row["s3q08"] = float(area_ha) * 10000
    row["lat_mod"] = float(lat)
    row["lon_mod"] = float(lon)
    for c in CATEGORICAL:
        if c == "s4q01b":
            row[c] = str(crop_code)
        else:
            row[c] = "MISSING"
    return pd.DataFrame([row], columns=REC_FEATURES)

def predict_crops(area_ha, lat, lon):
    crops, _ = load_data()
    yield_model, _, _ = load_models()
    codes = crops["s4q01b"].astype(str).tolist()
    results = []
    for code in codes:
        df_farm = build_farm(area_ha, lat, lon, code)
        pred_yield = yield_model.predict(df_farm)[0]
        results.append({
            "code": code,
            "crop": CROP_NAMES.get(str(code), f"Crop {code}"),
            "predicted_yield_kg_ha": max(0, float(pred_yield))
        })
    res_df = pd.DataFrame(results).sort_values(by="predicted_yield_kg_ha", ascending=False)
    return res_df

# Streamlit User Interface
st.title("🌾 AI Farmer Decision Support System")

st.sidebar.header("Farm Location & Parameters")
area_ha = st.sidebar.number_input("Farm Area (Hectares)", min_value=0.1, max_value=100.0, value=1.0)
lat = st.sidebar.number_input("Latitude", value=9.0)
lon = st.sidebar.number_input("Longitude", value=38.7)
ph = st.sidebar.slider("Soil pH Level", min_value=4.0, max_value=9.0, value=6.5, step=0.1)

if st.sidebar.button("Get Recommendations"):
    st.header("Top Recommended Crop")
    with st.spinner("Calculating predictions..."):
        crop_preds = predict_crops(area_ha, lat, lon)
        top_crop = crop_preds.iloc[0]
        st.success(f"**Recommended Crop:** {top_crop['crop']} (Estimated Yield: {top_crop['predicted_yield_kg_ha']:.2f} kg/ha)")

    st.subheader("Top 10 Suitable Crops")
    st.dataframe(crop_preds.head(10)[["crop", "predicted_yield_kg_ha"]], use_container_width=True)

    st.subheader("Fertilizer Advice")
    fert_type, fert_desc = fertilizer(ph)
    st.info(f"**Recommended Fertilizer:** {fert_type}\n\n*{fert_desc}*")

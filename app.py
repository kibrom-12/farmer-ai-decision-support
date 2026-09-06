import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from catboost import CatBoostRegressor, CatBoostClassifier

APP_DIR = Path(__file__).parent

st.set_page_config(page_title="AI Farmer Decision Support", page_icon="🌾", layout="wide")

# ============================================================
# CROP MAPPINGS & TARGET FOCUS
# ============================================================
CROP_NAMES = {
    "1.0": "Barley", "2.0": "Maize", "3.0": "Teff", "6.0": "Sorghum", "8.0": "Wheat",
    "10.0": "Cassava", "12.0": "Haricot Beans", "13.0": "Horse Beans",
    "19.0": "Red Kidney Beans", "24.0": "Ground Nuts", "26.0": "Rape Seed",
    "28.0": "Sunflower", "38.0": "Red Pepper", "42.0": "Bananas",
    "46.0": "Mangos", "47.0": "Oranges", "48.0": "Papaya", "55.0": "Garlic",
    "56.0": "Kale", "61.0": "Pumpkins", "62.0": "Sweet Potato", "71.0": "Chat",
    "72.0": "Coffee", "74.0": "Enset", "75.0": "Gesho", "76.0": "Sugar Cane",
    "84.0": "Avocados", "98.0": "Other Root Crops"
}

CROP_GROUPS = {
    "1.0": "Cereal", "2.0": "Cereal", "3.0": "Cereal", "6.0": "Cereal", "8.0": "Cereal",
    "10.0": "Root/Tuber", "62.0": "Root/Tuber", "98.0": "Root/Tuber",
    "12.0": "Legume", "13.0": "Legume", "19.0": "Legume", "24.0": "Legume",
    "26.0": "Oilseed", "28.0": "Oilseed",
    "38.0": "Vegetable", "55.0": "Vegetable", "56.0": "Vegetable", "61.0": "Vegetable",
    "42.0": "Fruit", "46.0": "Fruit", "47.0": "Fruit", "48.0": "Fruit", "84.0": "Fruit",
    "71.0": "Specialty",
    "72.0": "Perennial", "74.0": "Perennial", "75.0": "Perennial", "76.0": "Perennial"
}

# Explicit high-priority crop codes for Ethiopia cereal focus
PRIORITY_CROPS = ["1.0", "2.0", "3.0", "6.0", "8.0"]  # Barley, Maize, Teff, Sorghum, Wheat

# Optimal pH ranges (min_ph, max_ph) for pH penalty weighting
CROP_PH_RANGES = {
    "1.0": (6.0, 7.5), "2.0": (5.8, 7.0), "3.0": (5.5, 7.5), "6.0": (5.5, 7.5), "8.0": (6.0, 7.0),
    "10.0": (5.5, 6.5), "12.0": (6.0, 7.0), "13.0": (6.0, 7.2), "19.0": (6.0, 7.0),
    "24.0": (5.3, 6.5), "26.0": (5.8, 7.5), "28.0": (6.0, 7.2), "38.0": (6.0, 7.0),
    "42.0": (5.5, 6.5), "46.0": (5.5, 7.5), "47.0": (5.5, 6.5), "48.0": (6.0, 6.5),
    "55.0": (6.0, 7.0), "56.0": (6.0, 7.5), "61.0": (6.0, 7.5), "62.0": (5.5, 6.8),
    "71.0": (6.0, 7.0), "72.0": (5.0, 6.0), "74.0": (5.5, 6.5), "75.0": (6.0, 7.0),
    "76.0": (6.0, 7.5), "84.0": (5.5, 6.5), "98.0": (5.5, 6.5)
}

CROP_CODES = list(CROP_NAMES.keys())

REC_FEATURES = [
    "saq14", "saq01", "saq02", "saq03", "saq04", "saq05", "saq06", "saq07", "saq15",
    "s4q01b", "s3q02a", "s3q02b", "s3q03", "s3q04", "s3q05", "s3q07", "s3q08",
    "s3q12", "s3q16", "s3q28", "s3q35", "s3q36", "s3q38", "s3q40", "s3q42",
    "dist_road", "dist_market", "dist_popcenter", "ssa_aez09", "twi",
    "sq1", "sq2", "sq3", "sq4", "sq5", "sq6", "sq7",
    "af_bio_1", "af_bio_8", "af_bio_12", "af_bio_13", "af_bio_16",
    "slopepct", "srtm1k", "popdensity", "cropshare",
    "anntot_avg", "wetQ_avgstart", "wetQ_avg", "ndvi_avg", "lat_mod", "lon_mod"
]

CATEGORICAL_FEATURES = [
    "saq14", "saq01", "saq02", "saq06", "saq07", "saq15", "s4q01b",
    "s3q02b", "s3q03", "s3q04", "s3q05", "s3q07", "s3q12", "s3q16", "s3q35",
    "s3q36", "s3q38", "s3q40", "s3q42", "ssa_aez09", "sq1", "sq2", "sq3",
    "sq4", "sq5", "sq6", "sq7"
]

RAIN_FEATURES = [
    "rain_lag_1", "rain_lag_2", "rain_lag_3", "rain_lag_7", "rain_lag_14", "rain_lag_21", "rain_lag_28",
    "rain_roll_3", "rain_roll_7", "rain_roll_14", "rain_roll_28",
    "temp_lag_1", "temp_lag_7", "temp_lag_14", "dayofyear", "month", "sin_doy", "cos_doy"
]

# ============================================================
# LOAD MODELS & DATA
# ============================================================
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
def load_rainfall():
    return pd.read_csv(APP_DIR / "rainfall_history.csv")

# ============================================================
# PREDICTION LOGIC
# ============================================================
def calculate_ph_multiplier(crop_code, ph_val):
    min_ph, max_ph = CROP_PH_RANGES.get(crop_code, (5.5, 7.5))
    if min_ph <= ph_val <= max_ph:
        return 1.0
    elif ph_val < min_ph:
        diff = min_ph - ph_val
        return max(0.1, 1.0 - (diff * 0.35))
    else:
        diff = ph_val - max_ph
        return max(0.1, 1.0 - (diff * 0.35))

def build_farm(area_ha, latitude, longitude, crop_code):
    row = {f: 0.0 for f in REC_FEATURES}
    row["s4q01b"] = str(crop_code)
    row["s3q08"] = float(area_ha) * 10000.0
    row["lat_mod"] = float(latitude)
    row["lon_mod"] = float(longitude)
    
    for f in CATEGORICAL_FEATURES:
        row[f] = str(crop_code) if f == "s4q01b" else "0"
        
    df = pd.DataFrame([row], columns=REC_FEATURES)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    return df

def predict_crops(area_ha, latitude, longitude, ph_val, focus_cereals_only=True):
    yield_model, _, _ = load_models()
    results = []
    
    codes_to_eval = PRIORITY_CROPS if focus_cereals_only else CROP_CODES
    
    for code in codes_to_eval:
        farm = build_farm(area_ha, latitude, longitude, code)
        pred_log_yield = float(yield_model.predict(farm)[0])
        base_yield = max(0.0, float(np.expm1(pred_log_yield)))
        
        ph_mult = calculate_ph_multiplier(code, ph_val)
        adjusted_yield = base_yield * ph_mult
        
        results.append({
            "Crop": CROP_NAMES[code],
            "Crop Code": code,
            "Crop Group": CROP_GROUPS[code],
            "Predicted Yield (kg/ha)": adjusted_yield
        })
    df = pd.DataFrame(results)
    mn, mx = df["Predicted Yield (kg/ha)"].min(), df["Predicted Yield (kg/ha)"].max()
    df["Decision Degree (%)"] = ((df["Predicted Yield (kg/ha)"] - mn) / (mx - mn) * 100).clip(0, 100).round(1) if mx > mn else 100.0
    df = df.sort_values("Decision Degree (%)", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    return df

def forecast_rain(days=7):
    history = load_rainfall()
    _, rain_event_model, rain_amount_model = load_models()
    history["time"] = pd.to_datetime(history["time"])
    history = history.sort_values("time").reset_index(drop=True)
    
    rain_vals = history["rain_sum"].astype(float).tolist()
    temp_vals = history["temperature_2m_mean"].astype(float).tolist()
    last_date = history["time"].iloc[-1]
    
    avg_hist_rain = np.mean([r for r in rain_vals if r > 0]) if any(r > 0 for r in rain_vals) else 2.5
    
    forecasts = []
    for step in range(1, days + 1):
        fc_date = last_date + pd.Timedelta(days=step)
        
        def lag(v, n): return float(v[-n]) if len(v) >= n else 0.0
        def roll(v, n): return float(np.mean(v[-n:])) if len(v) >= n else float(np.mean(v))
        
        doy = fc_date.dayofyear
        x = pd.DataFrame([{
            "rain_lag_1": lag(rain_vals, 1), "rain_lag_2": lag(rain_vals, 2), "rain_lag_3": lag(rain_vals, 3),
            "rain_lag_7": lag(rain_vals, 7), "rain_lag_14": lag(rain_vals, 14), "rain_lag_21": lag(rain_vals, 21),
            "rain_lag_28": lag(rain_vals, 28), "rain_roll_3": roll(rain_vals, 3), "rain_roll_7": roll(rain_vals, 7),
            "rain_roll_14": roll(rain_vals, 14), "rain_roll_28": roll(rain_vals, 28),
            "temp_lag_1": lag(temp_vals, 1), "temp_lag_7": lag(temp_vals, 7), "temp_lag_14": lag(temp_vals, 14),
            "dayofyear": doy, "month": fc_date.month,
            "sin_doy": np.sin(2 * np.pi * doy / 365.25), "cos_doy": np.cos(2 * np.pi * doy / 365.25)
        }], columns=RAIN_FEATURES)
        
        prob = float(rain_event_model.predict_proba(x)[0, 1])
        pred_log = float(rain_amount_model.predict(x)[0])
        raw_pred = max(0.0, float(np.expm1(pred_log)))
        
        pred_rain = raw_pred if prob >= 0.3 else (avg_hist_rain * prob)
        
        rain_vals.append(pred_rain)
        temp_vals.append(lag(temp_vals, 1))
        forecasts.append({
            "Date": fc_date.strftime("%Y-%m-%d"),
            "Rain Probability (%)": round(prob * 100, 1),
            "Predicted Rain (mm)": round(pred_rain, 2)
        })
    return pd.DataFrame(forecasts)

def fertilizer_advice(ph_val):
    if ph_val < 5.5:
        return "Phosphorus-containing fertilizer", "Strongly acidic soil: acidity management should be considered using soil testing."
    elif ph_val < 6.5:
        return "Nitrogen + phosphorus fertilizer", "Acidic soil: nitrogen and phosphorus support recommended."
    elif ph_val <= 7.0:
        return "Balanced NPK fertilizer", "Near-neutral soil: balanced nutrient management is optimal."
    elif ph_val <= 7.5:
        return "Balanced NPK fertilizer", "Slightly alkaline soil: monitor micronutrient availability."
    else:
        return "Phosphorus or balanced fertilizer", "Alkaline soil: consult local agricultural guidance for optimal yield."

# ============================================================
# USER INTERFACE
# ============================================================
st.title("🌾 AI Farmer Decision Support System")

st.sidebar.header("Location & Administrative Inputs")
region = st.sidebar.text_input("Region", value="Oromia")
zone = st.sidebar.text_input("Zone", value="East Shewa")
woreda = st.sidebar.text_input("Woreda", value="Ada'a")

st.sidebar.header("Farm Coordinates & Soil")
area_ha = st.sidebar.number_input("Farm Area (Hectares)", min_value=0.1, max_value=100.0, value=1.0)
lat = st.sidebar.number_input("Latitude", value=8.54)
lon = st.sidebar.number_input("Longitude", value=38.98)
ph = st.sidebar.slider("Soil pH Level", min_value=4.0, max_value=9.0, value=6.5, step=0.1)

cereal_only = st.sidebar.checkbox("Focus Exclusively on Major Cereal Crops", value=True)

if st.sidebar.button("Run Analysis"):
    with st.spinner("Analyzing farm location data & running AI models..."):
        crop_df = predict_crops(area_ha, lat, lon, ph, focus_cereals_only=cereal_only)
        top_crop = crop_df.iloc[0]
        rain_df = forecast_rain(days=7)

        st.info(f"**Location Selected:** {woreda}, {zone}, {region} ({lat}° N, {lon}° E)")

        # 1. Recommended Crop
        st.header("🏆 Recommended Crop")
        st.success(f"**{top_crop['Crop']}** — Estimated Yield: **{top_crop['Predicted Yield (kg/ha)']:.2f} kg/ha** (Decision Degree: **{top_crop['Decision Degree (%)']:.1f}%**)")

        # 2. Major Crop Ranking Chart
        st.header("📊 Suitable Cereal Crops Ranking")
        fig_crops = px.bar(
            crop_df,
            x="Crop",
            y="Predicted Yield (kg/ha)",
            color="Decision Degree (%)",
            text_auto=".1f",
            title="Cereal Crop Yield Predictions (kg/ha)",
            color_continuous_scale="Greens"
        )
        fig_crops.update_layout(xaxis_title="Crop Name", yaxis_title="Yield (kg/ha)")
        st.plotly_chart(fig_crops, use_container_width=True)

        # 3. 7-Day Rainfall Forecast
        st.header("🌧️ 7-Day Rainfall Forecast")
        
        total_rain = rain_df["Predicted Rain (mm)"].sum()
        max_rain_row = rain_df.loc[rain_df["Predicted Rain (mm)"].idxmax()]
        avg_prob = rain_df["Rain Probability (%)"].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total 7-Day Rainfall", f"{total_rain:.1f} mm")
        col2.metric("Wettest Day Expected", f"{max_rain_row['Date']}", f"{max_rain_row['Predicted Rain (mm)']:.1f} mm")
        col3.metric("Average Rain Chance", f"{avg_prob:.0f}%")

        chart_col, table_col = st.columns([2, 1])

        with chart_col:
            fig_rain = go.Figure()
            fig_rain.add_trace(
                go.Bar(
                    x=rain_df["Date"],
                    y=rain_df["Predicted Rain (mm)"],
                    name="Rainfall Volume (mm)",
                    marker_color="#1f77b4"
                )
            )
            fig_rain.add_trace(
                go.Scatter(
                    x=rain_df["Date"],
                    y=rain_df["Rain Probability (%)"],
                    name="Chance of Rain (%)",
                    yaxis="y2",
                    mode="lines+markers",
                    line=dict(color="#ff7f0e", width=3)
                )
            )
            fig_rain.update_layout(
                title="Daily Expected Rainfall & Probability",
                xaxis_title="Date",
                yaxis=dict(title="Predicted Rain (mm)"),
                yaxis2=dict(title="Chance of Rain (%)", overlaying="y", side="right", range=[0, 100]),
                legend=dict(x=0.01, y=0.99),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_rain, use_container_width=True)

        with table_col:
            st.subheader("📋 Daily Breakdown")
            st.dataframe(
                rain_df[["Date", "Predicted Rain (mm)", "Rain Probability (%)"]],
                hide_index=True,
                use_container_width=True
            )

        # 4. Fertilizer Recommendation
        st.header("🧪 Fertilizer Advice (Based on Soil pH)")
        fert, desc = fertilizer_advice(ph)
        st.info(f"**Recommended Fertilizer:** {fert}\n\n*{desc}*")

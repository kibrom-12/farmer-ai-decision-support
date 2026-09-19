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
# IMPORTANT: These are the crop codes supported by the finalized yield model.
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
    "saq14", "saq01", "saq02", "saq03", "saq04", "saq05", "saq06", "saq07",
    "saq15", "s4q01b", "s3q02a", "s3q02b", "s3q03", "s3q04", "s3q05", "s3q07",
    "s3q08", "s3q12", "s3q16", "s3q28", "s3q35", "s3q36", "s3q38", "s3q40",
    "s3q42", "dist_road", "dist_market", "dist_popcenter", "ssa_aez09", "twi",
    "sq1", "sq2", "sq3", "sq4", "sq5", "sq6", "sq7", "af_bio_1", "af_bio_8",
    "af_bio_12", "af_bio_13", "af_bio_16", "slopepct", "srtm1k", "popdensity",
    "cropshare", "anntot_avg", "wetQ_avgstart", "wetQ_avg", "ndvi_avg",
    "lat_mod", "lon_mod"
]

# ============================================================
# CATEGORICAL FEATURES
# ============================================================
CATEGORICAL_FEATURES = [
    "saq14", "saq01", "saq02", "saq06", "saq07", "saq15", "s4q01b", "s3q02b",
    "s3q03", "s3q04", "s3q05", "s3q07", "s3q12", "s3q16", "s3q35", "s3q36",
    "s3q38", "s3q40", "s3q42", "ssa_aez09", "sq1", "sq2", "sq3", "sq4", "sq5",
    "sq6", "sq7"
]

# ============================================================
# RAINFALL FEATURES
# ============================================================
RAIN_FEATURES = [
    "rain_lag_1", "rain_lag_2", "rain_lag_3", "rain_lag_7", "rain_lag_14",
    "rain_lag_21", "rain_lag_28", "rain_roll_3", "rain_roll_7", "rain_roll_14",
    "rain_roll_28", "temp_lag_1", "temp_lag_7", "temp_lag_14", "dayofyear",
    "month", "sin_doy", "cos_doy"
]

# ============================================================
# LOAD AI MODELS
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

# ============================================================
# LOAD DATASETS
# ============================================================
@st.cache_data
def load_rainfall():
    file_path = APP_DIR / "rainfall_history.csv"
    if not file_path.exists():
        raise FileNotFoundError("rainfall_history.csv was not found in the GitHub repository.")
    return pd.read_csv(file_path)

@st.cache_data
def load_fertilizer_data():
    file_path = APP_DIR / "TAMASA_fertilizer_modeling_table.csv"
    if not file_path.exists():
        raise FileNotFoundError("TAMASA_fertilizer_modeling_table.csv was not found in the GitHub repository.")
    return pd.read_csv(file_path)

# ============================================================
# pH MULTIPLIER CALCULATOR
# ============================================================
def calculate_ph_multiplier(crop_code, ph_value):
    minimum_ph, maximum_ph = CROP_PH_RANGES.get(crop_code, (5.5, 7.5))
    if minimum_ph <= ph_value <= maximum_ph:
        return 1.0
    if ph_value < minimum_ph:
        difference = minimum_ph - ph_value
        return max(0.10, 1.0 - difference * 0.35)
    difference = ph_value - maximum_ph
    return max(0.10, 1.0 - difference * 0.35)

# ============================================================
# BUILD FARM INPUT DATAFRAME
# ============================================================
def build_farm(area_ha, latitude, longitude, crop_code):
    row = {feature: 0.0 for feature in REC_FEATURES}
    row["s4q01b"] = str(crop_code)
    row["s3q08"] = float(area_ha) * 10000.0
    row["lat_mod"] = float(latitude)
    row["lon_mod"] = float(longitude)

    for feature in CATEGORICAL_FEATURES:
        if feature == "s4q01b":
            row[feature] = str(crop_code)
        else:
            row[feature] = "0"

    farm_df = pd.DataFrame([row], columns=REC_FEATURES)
    for feature in CATEGORICAL_FEATURES:
        farm_df[feature] = farm_df[feature].fillna("MISSING").astype(str)
    return farm_df

# ============================================================
# CROP YIELD PREDICTION ENGINE
# ============================================================
def predict_crops(area_ha, latitude, longitude, ph_value, focus_cereals=False):
    yield_model, _, _ = load_models()
    results = []

    codes_to_use = CROP_CODES.copy()
    if focus_cereals:
        codes_to_use = [code for code in CROP_CODES if CROP_GROUPS.get(code) == "Cereal"]

    for crop_code in codes_to_use:
        farm_df = build_farm(area_ha=area_ha, latitude=latitude, longitude=longitude, crop_code=crop_code)
        predicted_log_yield = float(yield_model.predict(farm_df)[0])
        base_yield = max(0.0, float(np.expm1(predicted_log_yield)))
        ph_multiplier = calculate_ph_multiplier(crop_code, ph_value)
        adjusted_yield = base_yield * ph_multiplier

        results.append({
            "Crop": CROP_NAMES[crop_code],
            "Crop Code": crop_code,
            "Crop Group": CROP_GROUPS[crop_code],
            "Predicted Yield (kg/ha)": adjusted_yield
        })

    crop_df = pd.DataFrame(results)
    minimum_yield = crop_df["Predicted Yield (kg/ha)"].min()
    maximum_yield = crop_df["Predicted Yield (kg/ha)"].max()

    if maximum_yield > minimum_yield:
        crop_df["Decision Degree (%)"] = (
            (crop_df["Predicted Yield (kg/ha)"] - minimum_yield) / (maximum_yield - minimum_yield) * 100
        ).clip(0, 100)
    else:
        crop_df["Decision Degree (%)"] = 100.0

    crop_df["Decision Degree (%)"] = crop_df["Decision Degree (%)"].round(1)
    crop_df = crop_df.sort_values("Decision Degree (%)", ascending=False).reset_index(drop=True)
    crop_df["Rank"] = crop_df.index + 1
    return crop_df

# ============================================================
# RAINFALL FORECAST ENGINE
# ============================================================
def forecast_rain(days=7):
    history = load_rainfall()
    _, rain_event_model, rain_amount_model = load_models()

    history["time"] = pd.to_datetime(history["time"])
    history = history.sort_values("time").reset_index(drop=True)

    if "rain_sum" not in history.columns:
        raise ValueError("rainfall_history.csv does not contain required 'rain_sum' column.")
    if "temperature_2m_mean" not in history.columns:
        raise ValueError("rainfall_history.csv does not contain required 'temperature_2m_mean' column.")

    rain_values = history["rain_sum"].astype(float).tolist()
    temperature_values = history["temperature_2m_mean"].astype(float).tolist()
    last_date = history["time"].iloc[-1]

    positive_rain = [val for val in rain_values if val > 0]
    historical_rain_mean = float(np.mean(positive_rain)) if positive_rain else 2.5

    forecasts = []
    for step in range(1, days + 1):
        forecast_date = last_date + pd.Timedelta(days=step)

        def get_lag(values, lag_number):
            return float(values[-lag_number]) if len(values) >= lag_number else 0.0

        def get_rolling_mean(values, window):
            if len(values) >= window:
                return float(np.mean(values[-window:]))
            return float(np.mean(values)) if values else 0.0

        day_of_year = forecast_date.dayofyear
        features = {
            "rain_lag_1": get_lag(rain_values, 1),
            "rain_lag_2": get_lag(rain_values, 2),
            "rain_lag_3": get_lag(rain_values, 3),
            "rain_lag_7": get_lag(rain_values, 7),
            "rain_lag_14": get_lag(rain_values, 14),
            "rain_lag_21": get_lag(rain_values, 21),
            "rain_lag_28": get_lag(rain_values, 28),
            "rain_roll_3": get_rolling_mean(rain_values, 3),
            "rain_roll_7": get_rolling_mean(rain_values, 7),
            "rain_roll_14": get_rolling_mean(rain_values, 14),
            "rain_roll_28": get_rolling_mean(rain_values, 28),
            "temp_lag_1": get_lag(temperature_values, 1),
            "temp_lag_7": get_lag(temperature_values, 7),
            "temp_lag_14": get_lag(temperature_values, 14),
            "dayofyear": day_of_year,
            "month": forecast_date.month,
            "sin_doy": np.sin(2 * np.pi * day_of_year / 365.25),
            "cos_doy": np.cos(2 * np.pi * day_of_year / 365.25)
        }

        rain_input = pd.DataFrame([features], columns=RAIN_FEATURES)
        rain_probability = float(rain_event_model.predict_proba(rain_input)[0, 1])
        predicted_log_amount = float(rain_amount_model.predict(rain_input)[0])
        raw_rain = max(0.0, float(np.expm1(predicted_log_amount)))

        if rain_probability >= 0.30:
            predicted_rain = raw_rain
        else:
            predicted_rain = historical_rain_mean * rain_probability

        rain_values.append(predicted_rain)
        temperature_values.append(temperature_values[-1] if temperature_values else 20.0)

        forecasts.append({
            "Date": forecast_date.strftime("%Y-%m-%d"),
            "Rain Probability (%)": round(rain_probability * 100, 1),
            "Predicted Rain (mm)": round(predicted_rain, 2)
        })

    return pd.DataFrame(forecasts)

# ============================================================
# PAGE HEADER
# ============================================================
st.title("🌾 AI Farmer Decision Support System")
st.caption("AI crop recommendation • yield prediction • rainfall forecasting • fertilizer decision support")

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.header("📍 Farm Information")
region = st.sidebar.text_input("Region", value="Oromia")
zone = st.sidebar.text_input("Zone", value="East Shewa")
woreda = st.sidebar.text_input("Woreda", value="Ada'a")
area_ha = st.sidebar.number_input("Farm Area (hectares)", min_value=0.1, max_value=100.0, value=1.0, step=0.1)

latitude = st.sidebar.number_input("Latitude", min_value=3.0, max_value=15.0, value=8.54, step=0.01)
longitude = st.sidebar.number_input("Longitude", min_value=33.0, max_value=48.0, value=38.98, step=0.01)

st.sidebar.header("🧪 Soil Measurements")
ph_val = st.sidebar.slider("Soil pH", min_value=4.0, max_value=9.0, value=6.5, step=0.1)
total_n = st.sidebar.number_input("Total Nitrogen (N %)", min_value=0.01, max_value=2.0, value=0.15, step=0.01)
avail_p = st.sidebar.number_input("Available Phosphorus (P ppm)", min_value=0.5, max_value=100.0, value=12.0, step=0.5)

st.sidebar.caption(
    "The current fertilizer AI uses soil pH, total N, and available P. "
    "Soil K is omitted because complete soil-K measurements were not available in the TAMASA dataset."
)

focus_cereals = st.sidebar.checkbox("Focus Crop Recommendation on Cereals Only", value=True)

# ============================================================
# MAIN APPLICATION INTERFACE
# ============================================================
tab1, tab2, tab3 = st.tabs(["🌾 Crop Recommendation", "🌧️ 7-Day Rain Forecast", "🧪 Fertilizer Decision Support"])

with tab1:
    st.subheader("🌾 AI Crop Recommendation & Yield Ranking")
    if st.button("Run Crop Analysis", type="primary"):
        with st.spinner("Calculating optimal crop recommendations..."):
            crop_df = predict_crops(
                area_ha=area_ha,
                latitude=latitude,
                longitude=longitude,
                ph_value=ph_val,
                focus_cereals=focus_cereals
            )

            top_crop = crop_df.iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("Top Recommended Crop", top_crop["Crop"])
            col2.metric("Predicted Yield", f"{top_crop['Predicted Yield (kg/ha)']:.1f} kg/ha")
            col3.metric("Decision Match", f"{top_crop['Decision Degree (%)']}%")

            st.dataframe(
                crop_df[["Rank", "Crop", "Crop Group", "Predicted Yield (kg/ha)", "Decision Degree (%)"]],
                use_container_width=True
            )

            fig = px.bar(
                crop_df.head(10),
                x="Crop",
                y="Predicted Yield (kg/ha)",
                color="Decision Degree (%)",
                title="Top Recommended Crops by Yield Potential",
                labels={"Predicted Yield (kg/ha)": "Yield (kg/ha)"}
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🌧️ 7-Day Rainfall Forecast")
    if st.button("Generate Weather Forecast"):
        with st.spinner("Fetching precipitation probabilities..."):
            forecast_df = forecast_rain(days=7)
            st.dataframe(forecast_df, use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=forecast_df["Date"], y=forecast_df["Predicted Rain (mm)"], name="Rainfall (mm)"))
            fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Rain Probability (%)"], name="Rain Prob (%)", yaxis="y2"))

            fig.update_layout(
                title="7-Day Precipitation & Probability Trend",
                yaxis=dict(title="Predicted Rain (mm)"),
                yaxis2=dict(title="Probability (%)", overlaying="y", side="right")
            )
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("🧪 Fertilizer Decision Engine")
    st.markdown(
        """
        **Current fertilizer AI inputs**
        - Soil pH
        - Total Nitrogen (N)
        - Available Phosphorus (P)

        **Fertilizer decision outputs**
        - Recommended fertilizer strategy
        - Estimated yield impact
        - Treatment comparisons
        """
    )
    if st.button("Calculate Fertilizer Strategy"):
        fert_data = load_fertilizer_data()
        recommendation = recommend_fertilizer(
            ph_val=ph_val,
            total_n=total_n,
            avail_p=avail_p,
            df_dataset=fert_data
        )
        st.json(recommendation)

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
# CROP INFORMATION
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
    "1.0": "cereal",
    "2.0": "cereal",
    "6.0": "cereal",
    "8.0": "cereal",
    "10.0": "root_tuber",
    "62.0": "root_tuber",
    "98.0": "root_tuber",
    "12.0": "legume",
    "13.0": "legume",
    "19.0": "legume",
    "24.0": "legume",
    "26.0": "oilseed",
    "28.0": "oilseed",
    "38.0": "vegetable",
    "55.0": "vegetable",
    "56.0": "vegetable",
    "61.0": "vegetable",
    "42.0": "fruit",
    "46.0": "fruit",
    "47.0": "fruit",
    "48.0": "fruit",
    "84.0": "fruit",
    "71.0": "specialty",
    "72.0": "perennial",
    "74.0": "perennial",
    "75.0": "perennial",
    "76.0": "perennial"
}

CROP_CODES = list(CROP_NAMES.keys())

# ============================================================
# YIELD MODEL FEATURES
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
# BUILD FARM INPUT
# ============================================================

def build_farm(area_ha, latitude, longitude, crop_code):

    row = {}

    for feature in REC_FEATURES:
        row[feature] = 0

    # Crop code
    row["s4q01b"] = str(crop_code)

    # Field area
    row["s3q08"] = float(area_ha) * 10000

    # Location
    row["lat_mod"] = float(latitude)
    row["lon_mod"] = float(longitude)

    # CatBoost categorical variables must be strings
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
# CROP PREDICTION
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
            "Code": code,
            "Group": CROP_GROUPS[code],
            "Predicted yield (kg/ha)": predicted_yield
        })

    result = pd.DataFrame(results)

    minimum = result[
        "Predicted yield (kg/ha)"
    ].min()

    maximum = result[
        "Predicted yield (kg/ha)"
    ].max()

    if maximum > minimum:

        result["Yield score"] = (
            (
                result["Predicted yield (kg/ha)"]
                - minimum
            )
            /
            (maximum - minimum)
        ) * 100

    else:

        result["Yield score"] = 100

    # Conservative decision score
    # Reliability is fixed here because the deployment
    # reliability table is not required for prediction.
    result["Decision score"] = (
        0.65 * result["Yield score"]
        +
        0.35 * 50
    )

    result = result.sort_values(
        "Decision score",
        ascending=False
    ).reset_index(drop=True)

    result["Rank"] = (
        np.arange(len(result)) + 1
    )

    return result


# ============================================================
# RAINFALL FORECAST
# ============================================================

def forecast_rain(days=7):

    history = load_rainfall()

    _, rain_event_model, rain_amount_model = load_models()

    history["time"] = pd.to_datetime(
        history["time"]
    )

    history = history.sort_values(
        "time"
    ).reset_index(drop=True)

    rain_values = (
        history["rain_sum"]
        .astype(float)
        .tolist()
    )

    temp_values = (
        history["temperature_2m_mean"]
        .astype(float)
        .tolist()
    )

    last_date = history["time"].iloc[-1]

    forecasts = []

    for step in range(1, days + 1):

        forecast_date = (
            last_date
            +
            pd.Timedelta(days=step)
        )

        def lag(values, n):

            if len(values) >= n:
                return float(values[-n])

            return 0.0

        def rolling(values, n):

            if len(values) >= n:
                return float(
                    np.mean(values[-n:])
                )

            return float(
                np.mean(values)
            )

        doy = forecast_date.dayofyear

        x = pd.DataFrame([{

            "rain_lag_1":
                lag(rain_values, 1),

            "rain_lag_2":
                lag(rain_values, 2),

            "rain_lag_3":
                lag(rain_values, 3),

            "rain_lag_7":
                lag(rain_values, 7),

            "rain_lag_14":
                lag(rain_values, 14),

            "rain_lag_21":
                lag(rain_values, 21),

            "rain_lag_28":
                lag(rain_values, 28),

            "rain_roll_3":
                rolling(rain_values, 3),

            "rain_roll_7":
                rolling(rain_values, 7),

            "rain_roll_14":
                rolling(rain_values, 14),

            "rain_roll_28":
                rolling(rain_values, 28),

            "temp_lag_1":
                lag(temp_values, 1),

            "temp_lag_7":
                lag(temp_values, 7),

            "temp_lag_14":
                lag(temp_values, 14),

            "day_of_year":
                doy,

            "month":
                forecast_date.month,

            "sin_doy":
                np.sin(
                    2 * np.pi * doy / 365.25
                ),

            "cos_doy":
                np.cos(
                    2 * np.pi * doy / 365.25
                )

        }], columns=RAIN_FEATURES)

        rain_probability = float(
            rain_event_model.predict_proba(x)[0, 1]
        )

        predicted_log_rain = float(
            rain_amount_model.predict(x)[0]
        )

        predicted_rain = max(
            0.0,
            float(
                np.expm1(predicted_log_rain)
            )
        )

        if rain_probability < 0.5:
            predicted_rain = 0.0

        rain_values.append(
            predicted_rain
        )

        temp_values.append(
            lag(temp_values, 1)
        )

        forecasts.append({

            "Date":
                forecast_date.strftime(
                    "%Y-%m-%d"
                ),

            "Rain probability":
                round(
                    rain_probability,
                    3
                ),

            "Predicted rain (mm)":
                round(
                    predicted_rain,
                    2
                )
        })

    return pd.DataFrame(forecasts)


# ============================================================
# FERTILIZER — PH ONLY
# ============================================================

def fertilizer_recommendation(ph):

    if ph < 5.5:

        return (
            "Phosphorus-containing fertilizer",
            "The soil is strongly acidic. "
            "Soil-acidity management should be considered "
            "using soil testing and local extension advice."
        )

    elif ph < 6.5:

        return (
            "Nitrogen + phosphorus fertilizer",
            "The soil is acidic. Nitrogen and phosphorus "
            "support may be appropriate."
        )

    elif ph <= 7.0:

        return (
            "Balanced NPK/compound fertilizer",
            "The soil is near neutral. Balanced nutrient "
            "management is recommended."
        )

    elif ph <= 7.5:

        return (
            "Balanced NPK/compound fertilizer",
            "The soil is slightly alkaline. "
            "Monitor nutrient availability."
        )

    else:

        return (
            "Phosphorus-containing or balanced fertilizer",
            "The soil is alkaline and nutrient availability "
            "may be affected. Use soil-test/local guidance."
        )


# ============================================================
# USER INTERFACE
# ============================================================

st.title(
    "🌾 AI-Powered Farmer Decision Support"
)

st.write(
    "Crop recommendation • rainfall forecasting • "
    "fertilizer guidance • planting advice"
)

st.divider()

# ------------------------------------------------------------
# LOCATION AND FARM INPUTS
# ------------------------------------------------------------

st.header("📍 Farm Location & Parameters")

col1, col2 = st.columns(2)

with col1:

    region = st.text_input(
        "Region",
        value="Tigray"
    )

    zone = st.text_input(
        "Zone",
        value="Mekelle"
    )

    woreda = st.text_input(
        "Woreda",
        value="Mekelle"
    )

with col2:

    area_ha = st.number_input(
        "Farm Area (Hectares)",
        min_value=0.01,
        max_value=10000.0,
        value=1.0,
        step=0.1
    )

    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=14.0,
        step=0.01
    )

    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=38.0,
        step=0.01
    )

soil_ph = st.slider(
    "🧪 Soil pH Level",
    min_value=3.0,
    max_value=10.0,
    value=6.5,
    step=0.1
)

generate = st.button(
    "🚀 Get Recommendations",
    type="primary",
    use_container_width=True
)

# ============================================================
# RUN SYSTEM
# ============================================================

if generate:

    try:

        with st.spinner(
            "Running AI decision support system..."
        ):

            ranking = predict_crops(
                area_ha,
                latitude,
                longitude
            )

            rainfall = forecast_rain(7)

        # ----------------------------------------------------
        # FINAL CROP
        # ----------------------------------------------------

        top_crop = ranking.iloc[0]

        crop_name = top_crop["Crop"]
        predicted_yield = (
            top_crop[
                "Predicted yield (kg/ha)"
            ]
        )

        # ----------------------------------------------------
        # RAINFALL
        # ----------------------------------------------------

        total_rain = float(
            rainfall[
                "Predicted rain (mm)"
            ].sum()
        )

        effective_days = int(
            (
                rainfall[
                    "Predicted rain (mm)"
                ] >= 3.0
            ).sum()
        )

        if (
            total_rain >= 15
            and effective_days >= 2
        ):

            planting_decision = (
                "Planting window favorable"
            )

        elif (
            total_rain >= 5
            and effective_days >= 1
        ):

            planting_decision = (
                "Prepare and monitor rainfall"
            )

        else:

            planting_decision = (
                "Wait for more effective rainfall

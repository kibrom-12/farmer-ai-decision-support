import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from catboost import CatBoostRegressor, CatBoostClassifier

# ============================================================
# APP CONFIG
# ============================================================

APP_DIR = Path(__file__).parent

st.set_page_config(
    page_title="AI Farmer Decision Support",
    page_icon="🌾",
    layout="wide"
)

# ============================================================
# CROP MAPPINGS
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
    "98.0": "Other Root Crops"
}

CROP_GROUPS = {
    "1.0": "Cereal",
    "2.0": "Cereal",
    "3.0": "Cereal",
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
# OPTIMAL pH RANGES
# ============================================================

CROP_PH_RANGES = {
    "1.0": (6.0, 7.5),
    "2.0": (5.8, 7.0),
    "3.0": (5.5, 7.5),
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
    "sq7"
]

# ============================================================
# RAINFALL MODEL FEATURES
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
    "dayofyear",
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
# pH MULTIPLIER
# ============================================================

def calculate_ph_multiplier(
    crop_code,
    ph_val
):

    min_ph, max_ph = CROP_PH_RANGES.get(
        crop_code,
        (5.5, 7.5)
    )

    if min_ph <= ph_val <= max_ph:

        return 1.0

    elif ph_val < min_ph:

        diff = min_ph - ph_val

        return max(
            0.1,
            1.0 - (diff * 0.35)
        )

    else:

        diff = ph_val - max_ph

        return max(
            0.1,
            1.0 - (diff * 0.35)
        )

# ============================================================
# BUILD FARM INPUT
# ============================================================

def build_farm(
    area_ha,
    latitude,
    longitude,
    crop_code
):

    row = {
        feature: 0.0
        for feature in REC_FEATURES
    }

    row["s4q01b"] = str(crop_code)

    row["s3q08"] = (
        float(area_ha) * 10000.0
    )

    row["lat_mod"] = float(latitude)

    row["lon_mod"] = float(longitude)

    for feature in CATEGORICAL_FEATURES:

        if feature == "s4q01b":

            row[feature] = str(crop_code)

        else:

            row[feature] = "0"

    df = pd.DataFrame(
        [row],
        columns=REC_FEATURES
    )

    for col in CATEGORICAL_FEATURES:

        df[col] = df[col].astype(str)

    return df

# ============================================================
# CROP PREDICTION
# ============================================================

def predict_crops(
    area_ha,
    latitude,
    longitude,
    ph_val
):

    yield_model, _, _ = load_models()

    results = []

    for code in CROP_CODES:

        farm = build_farm(
            area_ha,
            latitude,
            longitude,
            code
        )

        pred_log_yield = float(
            yield_model.predict(farm)[0]
        )

        base_yield = max(
            0.0,
            float(
                np.expm1(
                    pred_log_yield
                )
            )
        )

        ph_multiplier = calculate_ph_multiplier(
            code,
            ph_val
        )

        adjusted_yield = (
            base_yield *
            ph_multiplier
        )

        results.append({

            "Crop": CROP_NAMES[code],

            "Crop Code": code,

            "Crop Group": CROP_GROUPS[code],

            "Predicted Yield (kg/ha)":
                adjusted_yield

        })

    df = pd.DataFrame(results)

    minimum = df[
        "Predicted Yield (kg/ha)"
    ].min()

    maximum = df[
        "Predicted Yield (kg/ha)"
    ].max()

    if maximum > minimum:

        df[
            "Decision Degree (%)"
        ] = (
            (
                df["Predicted Yield (kg/ha)"]
                - minimum
            )
            /
            (maximum - minimum)
            * 100
        ).clip(
            0,
            100
        ).round(1)

    else:

        df[
            "Decision Degree (%)"
        ] = 100.0

    df = df.sort_values(
        "Decision Degree (%)",
        ascending=False
    ).reset_index(
        drop=True
    )

    df["Rank"] = (
        df.index + 1
    )

    return df

# ============================================================
# 7-DAY AI RAINFALL FORECAST
# ============================================================

def forecast_rain(days=7):

    history = load_rainfall().copy()

    _, rain_event_model, rain_amount_model = (
        load_models()
    )

    history["time"] = pd.to_datetime(
        history["time"]
    )

    history = history.sort_values(
        "time"
    ).reset_index(
        drop=True
    )

    rain_vals = (
        history["rain_sum"]
        .astype(float)
        .tolist()
    )

    temp_vals = (
        history["temperature_2m_mean"]
        .astype(float)
        .tolist()
    )

    # ========================================================
    # IMPORTANT UPDATE:
    # Forecast DISPLAY now starts from today's date.
    # ========================================================

    last_date = pd.Timestamp.today().normalize()

    avg_hist_rain = (
        np.mean(
            [
                r for r in rain_vals
                if r > 0
            ]
        )
        if any(r > 0 for r in rain_vals)
        else 2.5
    )

    forecasts = []

    for step in range(
        0,
        days
    ):

        forecast_date = (
            last_date
            + pd.Timedelta(days=step)
        )

        def lag(values, n):

            if len(values) >= n:

                return float(
                    values[-n]
                )

            return 0.0

        def rolling_mean(
            values,
            n
        ):

            if len(values) >= n:

                return float(
                    np.mean(
                        values[-n:]
                    )
                )

            return float(
                np.mean(values)
            )

        doy = forecast_date.dayofyear

        x = pd.DataFrame(
            [{
                "rain_lag_1":
                    lag(rain_vals, 1),

                "rain_lag_2":
                    lag(rain_vals, 2),

                "rain_lag_3":
                    lag(rain_vals, 3),

                "rain_lag_7":
                    lag(rain_vals, 7),

                "rain_lag_14":
                    lag(rain_vals, 14),

                "rain_lag_21":
                    lag(rain_vals, 21),

                "rain_lag_28":
                    lag(rain_vals, 28),

                "rain_roll_3":
                    rolling_mean(rain_vals, 3),

                "rain_roll_7":
                    rolling_mean(rain_vals, 7),

                "rain_roll_14":
                    rolling_mean(rain_vals, 14),

                "rain_roll_28":
                    rolling_mean(rain_vals, 28),

                "temp_lag_1":
                    lag(temp_vals, 1),

                "temp_lag_7":
                    lag(temp_vals, 7),

                "temp_lag_14":
                    lag(temp_vals, 14),

                "dayofyear":
                    doy,

                "month":
                    forecast_date.month,

                "sin_doy":
                    np.sin(
                        2
                        * np.pi
                        * doy
                        / 365.25
                    ),

                "cos_doy":
                    np.cos(
                        2
                        * np.pi
                        * doy
                        / 365.25
                    )
            }],
            columns=RAIN_FEATURES
        )

        probability = float(
            rain_event_model
            .predict_proba(x)[0, 1]
        )

        predicted_log = float(
            rain_amount_model
            .predict(x)[0]
        )

        raw_prediction = max(
            0.0,
            float(
                np.expm1(
                    predicted_log
                )
            )
        )

        if probability >= 0.30:

            predicted_rain = raw_prediction

        else:

            predicted_rain = (
                avg_hist_rain
                * probability
            )

        rain_vals.append(
            predicted_rain
        )

        temp_vals.append(
            lag(temp_vals, 1)
        )

        forecasts.append({

            "Date":
                forecast_date.strftime(
                    "%Y-%m-%d"
                ),

            "Rain Probability (%)":
                round(
                    probability * 100,
                    1
                ),

            "Predicted Rain (mm)":
                round(
                    predicted_rain,
                    2
                )
        })

    return pd.DataFrame(
        forecasts
    )

# ============================================================
# TEMPORARY pH-BASED FERTILIZER GUIDANCE
# ============================================================

def fertilizer_advice(ph_val):

    if ph_val < 5.5:

        return (
            "Phosphorus-containing fertilizer",
            "Strongly acidic soil: acidity management "
            "should be considered using soil testing."
        )

    elif ph_val < 6.5:

        return (
            "Nitrogen + phosphorus fertilizer",
            "Acidic soil: nitrogen and phosphorus "
            "support recommended."
        )

    elif ph_val <= 7.0:

        return (
            "Balanced NPK fertilizer",
            "Near-neutral soil: balanced nutrient "
            "management is optimal."
        )

    elif ph_val <= 7.5:

        return (
            "Balanced NPK fertilizer",
            "Slightly alkaline soil: monitor "
            "micronutrient availability."
        )

    else:

        return (
            "Phosphorus or balanced fertilizer",
            "Alkaline soil: consult local agricultural "
            "guidance for optimal yield."
        )

# ============================================================
# USER INTERFACE
# ============================================================

st.title(
    "🌾 AI Farmer Decision Support System"
)

st.sidebar.header(
    "Location & Administrative Inputs"
)

region = st.sidebar.text_input(
    "Region",
    value="Oromia"
)

zone = st.sidebar.text_input(
    "Zone",
    value="East Shewa"
)

woreda = st.sidebar.text_input(
    "Woreda",
    value="Ada'a"
)

# ============================================================
# FARM INPUTS
# ============================================================

st.sidebar.header(
    "Farm Coordinates & Soil"
)

area_ha = st.sidebar.number_input(
    "Farm Area (Hectares)",
    min_value=0.1,
    max_value=100.0,
    value=1.0
)

lat = st.sidebar.number_input(
    "Latitude",
    value=8.54
)

lon = st.sidebar.number_input(
    "Longitude",
    value=38.98
)

ph = st.sidebar.slider(
    "Soil pH Level",
    min_value=4.0,
    max_value=9.0,
    value=6.5,
    step=0.1
)

# ============================================================
# CEREAL FOCUS
# ============================================================

st.sidebar.header(
    "🌾 Crop Focus"
)

focus_cereals = st.sidebar.checkbox(
    "🌾 Focus on cereal crops",
    value=False,
    help=(
        "When selected, the recommendation is "
        "limited to cereal crops such as "
        "Sorghum, Maize, Wheat, Barley and Teff."
    )
)

# ============================================================
# SOIL NPK
# ============================================================

st.sidebar.header(
    "🧪 Soil NPK Measurements"
)

soil_n = st.sidebar.number_input(
    "Nitrogen (N)",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help=(
        "Enter a laboratory or calibrated "
        "sensor measurement."
    )
)

soil_p = st.sidebar.number_input(
    "Phosphorus (P)",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help=(
        "Enter a laboratory or calibrated "
        "sensor measurement."
    )
)

soil_k = st.sidebar.number_input(
    "Potassium (K)",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help=(
        "Enter a laboratory or calibrated "
        "sensor measurement."
    )
)

st.sidebar.caption(
    "N/P/K values are collected as soil inputs. "
    "The validated fertilizer AI will be connected "
    "after the Ethiopian soil training data is prepared."
)

# ============================================================
# RUN ANALYSIS
# ============================================================

if st.sidebar.button(
    "Run Analysis"
):

    with st.spinner(
        "Analyzing farm location data & "
        "running AI models..."
    ):

        # ----------------------------------------------------
        # CROP PREDICTION
        # ----------------------------------------------------

        crop_df = predict_crops(
            area_ha,
            lat,
            lon,
            ph
        )

        # ----------------------------------------------------
        # CEREAL FILTER
        # ----------------------------------------------------

        if focus_cereals:

            crop_df = crop_df[
                crop_df["Crop Group"]
                == "Cereal"
            ].copy()

            crop_df = crop_df.sort_values(
                "Decision Degree (%)",
                ascending=False
            ).reset_index(
                drop=True
            )

            crop_df["Rank"] = (
                crop_df.index + 1
            )

        # ----------------------------------------------------
        # TOP CROP
        # ----------------------------------------------------

        top_crop = crop_df.iloc[0]

        # ----------------------------------------------------
        # RAINFALL
        # ----------------------------------------------------

        rain_df = forecast_rain(
            days=7
        )

        # ====================================================
        # LOCATION
        # ====================================================

        st.info(
            f"**Location Selected:** "
            f"{woreda}, {zone}, {region} "
            f"({lat}° N, {lon}° E)"
        )

        # ====================================================
        # 1. RECOMMENDED CROP
        # ====================================================

        st.header(
            "🏆 Recommended Crop"
        )

        if focus_cereals:

            st.caption(
                "🌾 Cereal focus is ON — "
                "recommendation is restricted to cereal crops."
            )

        st.success(
            f"**{top_crop['Crop']}** — "
            f"Estimated Yield: "
            f"**{top_crop['Predicted Yield (kg/ha)']:.2f} kg/ha** "
            f"(Decision Degree: "
            f"**{top_crop['Decision Degree (%)']:.1f}%**)"
        )

        # ====================================================
        # 2. TOP 10 CROPS
        # ====================================================

        st.header(
            "📊 Top 10 Suitable Crops"
        )

        top_10 = crop_df.head(10)

        fig_crops = px.bar(
            top_10,
            x="Crop",
            y="Predicted Yield (kg/ha)",
            color="Decision Degree (%)",
            text_auto=".1f",
            title=(
                "Top 10 Crop Yield Predictions (kg/ha)"
            ),
            color_continuous_scale="Greens"
        )

        fig_crops.update_layout(
            xaxis_title="Crop Name",
            yaxis_title="Yield (kg/ha)"
        )

        st.plotly_chart(
            fig_crops,
            use_container_width=True
        )

        st.dataframe(
            top_10[
                [
                    "Rank",
                    "Crop",
                    "Crop Group",
                    "Predicted Yield (kg/ha)",
                    "Decision Degree (%)"
                ]
            ],
            hide_index=True,
            use_container_width=True
        )

        # ====================================================
        # 3. 7-DAY RAINFALL
        # ====================================================

        st.header(
            "🌧️ 7-Day Rainfall Forecast"
        )

        total_rain = (
            rain_df[
                "Predicted Rain (mm)"
            ].sum()
        )

        max_rain_row = rain_df.loc[
            rain_df[
                "Predicted Rain (mm)"
            ].idxmax()
        ]

        avg_probability = (
            rain_df[
                "Rain Probability (%)"
            ].mean()
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total 7-Day Rainfall",
            f"{total_rain:.1f} mm"
        )

        col2.metric(
            "Wettest Day Expected",
            f"{max_rain_row['Date']}",
            f"{max_rain_row['Predicted Rain (mm)']:.1f} mm"
        )

        col3.metric(
            "Average Rain Chance",
            f"{avg_probability:.0f}%"
        )

        chart_col, table_col = st.columns(
            [2, 1]
        )

        # ----------------------------------------------------
        # RAIN CHART
        # ----------------------------------------------------

        with chart_col:

            fig_rain = go.Figure()

            fig_rain.add_trace(
                go.Bar(
                    x=rain_df["Date"],
                    y=rain_df[
                        "Predicted Rain (mm)"
                    ],
                    name="Rainfall Volume (mm)",
                    marker_color="#1f77b4"
                )
            )

            fig_rain.add_trace(
                go.Scatter(
                    x=rain_df["Date"],
                    y=rain_df[
                        "Rain Probability (%)"
                    ],
                    name="Chance of Rain (%)",
                    yaxis="y2",
                    mode="lines+markers",
                    line=dict(
                        color="#ff7f0e",
                        width=3
                    )
                )
            )

            fig_rain.update_layout(

                title=(
                    "Daily Expected Rainfall "
                    "& Probability"
                ),

                xaxis_title="Date",

                yaxis=dict(
                    title="Predicted Rain (mm)"
                ),

                yaxis2=dict(
                    title="Chance of Rain (%)",
                    overlaying="y",
                    side="right",
                    range=[0, 100]
                ),

                legend=dict(
                    x=0.01,
                    y=0.99
                ),

                margin=dict(
                    l=20,
                    r=20,
                    t=40,
                    b=20
                )
            )

            st.plotly_chart(
                fig_rain,
                use_container_width=True
            )

        # ----------------------------------------------------
        # RAIN TABLE
        # ----------------------------------------------------

        with table_col:

            st.subheader(
                "📋 Daily Breakdown"
            )

            st.dataframe(
                rain_df[
                    [
                        "Date",
                        "Predicted Rain (mm)",
                        "Rain Probability (%)"
                    ]
                ],
                hide_index=True,
                use_container_width=True
            )

        # ====================================================
        # 4. RAINFALL RISK
        # ====================================================

        st.header(
            "⚠️ Agricultural Rainfall Risk"
        )

        rainy_days = (
            rain_df[
                "Predicted Rain (mm)"
            ] > 1
        ).sum()

        heavy_days = (
            rain_df[
                "Predicted Rain (mm)"
            ] >= 10
        ).sum()

        if total_rain < 5:

            dryness_risk = 80

        elif total_rain < 15:

            dryness_risk = 50

        else:

            dryness_risk = 20

        if heavy_days >= 2:

            excess_rain_risk = 80

        elif heavy_days == 1:

            excess_rain_risk = 50

        else:

            excess_rain_risk = 20

        max_probability = (
            rain_df[
                "Rain Probability (%)"
            ].max()
        )

        if max_probability >= 75:

            probability_risk = 20

        elif max_probability >= 50:

            probability_risk = 40

        else:

            probability_risk = 60

        rainfall_risk_score = (
            0.45 * dryness_risk
            + 0.35 * excess_rain_risk
            + 0.20 * probability_risk
        )

        if rainfall_risk_score < 35:

            rainfall_risk_level = "Low"

        elif rainfall_risk_score < 55:

            rainfall_risk_level = "Moderate"

        elif rainfall_risk_score < 75:

            rainfall_risk_level = "High"

        else:

            rainfall_risk_level = "Very High"

        risk_col1, risk_col2, risk_col3 = st.columns(3)

        risk_col1.metric(
            "Rainfall Risk Score",
            f"{rainfall_risk_score:.1f}/100"
        )

        risk_col2.metric(
            "Risk Level",
            rainfall_risk_level
        )

        risk_col3.metric(
            "Rainy Days",
            int(rainy_days)
        )

        st.write(
            f"**Heavy-rain days:** {int(heavy_days)}"
        )

        if rainfall_risk_level == "Very High":

            st.error(
                "Very high rainfall-related risk. "
                "Review field drainage and planting decisions."
            )

        elif rainfall_risk_level == "High":

            st.warning(
                "High rainfall-related risk. "
                "Monitor rainfall closely."
            )

        elif rainfall_risk_level == "Moderate":

            st.info(
                "Moderate rainfall-related risk."
            )

        else:

            st.success(
                "Low rainfall-related risk."
            )

        # ====================================================
        # 5. SOIL NUTRIENT & FERTILIZER
        # ====================================================

        st.header(
            "🧪 Soil Nutrient & Fertilizer Decision Support"
        )

        st.write(
            "Current soil measurements entered "
            "for this analysis:"
        )

        npk_col1, npk_col2, npk_col3, npk_col4 = (
            st.columns(4)
        )

        with npk_col1:

            st.metric(
                "Soil pH",
                f"{ph:.1f}"
            )

        with npk_col2:

            st.metric(
                "Nitrogen (N)",
                f"{soil_n:.1f}"
            )

        with npk_col3:

            st.metric(
                "Phosphorus (P)",
                f"{soil_p:.1f}"
            )

        with npk_col4:

            st.metric(
                "Potassium (K)",
                f"{soil_k:.1f}"
            )

        # ====================================================
        # FERTILIZER AI
        # ====================================================

        st.markdown(
            "### 🤖 Fertilizer AI"
        )

        st.info(
            "The N/P/K fertilizer recommendation model "
            "is currently being prepared from Ethiopian "
            "soil data. Exact fertilizer recommendations "
            "will be generated after the soil N/P/K "
            "training data and appropriate agronomic "
            "reference labels are validated."
        )

        # ====================================================
        # TEMPORARY pH GUIDANCE
        # ====================================================

        st.markdown(
            "### 📌 Preliminary pH Guidance"
        )

        fertilizer, description = (
            fertilizer_advice(ph)
        )

        st.warning(
            f"**Current pH-based guidance:** "
            f"{fertilizer}\n\n"
            f"{description}\n\n"
            "⚠️ pH alone cannot determine soil N, P "
            "or K deficiency. This is temporary "
            "decision support and is not the final "
            "N/P/K fertilizer AI."
        )

        # ====================================================
        # 6. WHAT-IF pH ANALYSIS
        # ====================================================

        st.header(
            "🔬 What-If Analysis"
        )

        what_if_ph = st.slider(
            "Change soil pH to see how crop ranking changes",
            min_value=4.0,
            max_value=9.0,
            value=float(ph),
            step=0.1
        )

        what_if_df = predict_crops(
            area_ha,
            lat,
            lon,
            what_if_ph
        )

        if focus_cereals:

            what_if_df = what_if_df[
                what_if_df["Crop Group"]
                == "Cereal"
            ].copy()

            what_if_df = (
                what_if_df
                .sort_values(
                    "Decision Degree (%)",
                    ascending=False
                )
                .reset_index(drop=True)
            )

            what_if_df["Rank"] = (
                what_if_df.index + 1
            )

        what_if_top = (
            what_if_df.iloc[0]
        )

        st.success(
            f"With soil pH = **{what_if_ph:.1f}**, "
            f"the highest-ranked crop is "
            f"**{what_if_top['Crop']}**."
        )

        # ====================================================
        # 7. MODEL INFORMATION
        # ====================================================

        st.header(
            "🤖 AI Model Information"
        )

        st.write(
            "The crop recommendation component uses "
            "a CatBoost yield prediction model. "
            "Crop ranking combines predicted yield "
            "with crop-specific pH suitability."
        )

        st.write(
            "The rainfall component uses separate "
            "CatBoost models for rainfall-event "
            "probability and rainfall amount."
        )

        st.caption(
            "The current rainfall forecast is based "
            "on the trained historical rainfall model. "
            "Changing the displayed forecast date does "
            "not make the underlying weather data live."
        )

        # ====================================================
        # 8. FUTURE IoT
        # ====================================================

        st.header(
            "🔌 Future IoT Integration"
        )

        st.write(
            "NPK and pH measurements can later be "
            "supplied by calibrated soil sensors "
            "through an IoT device such as an ESP32."
        )

        st.write(
            "The same measurements will feed the "
            "validated fertilizer model once the "
            "Ethiopian soil dataset and agronomic "
            "reference labels are available."
        )

        # ====================================================
        # 9. SYSTEM DISCLAIMER
        # ====================================================

        st.header(
            "⚠️ Decision Support Disclaimer"
        )

        st.caption(
            "AI predictions are estimates and should "
            "not be treated as guaranteed agricultural "
            "outcomes. Final fertilizer application "
            "rates should be based on validated soil "
            "testing and appropriate local agronomic "
            "guidance."
        )

else:

    st.info(
        "👈 Enter the farmer/farm information "
        "in the sidebar and click **Run Analysis** "
        "to generate the AI decision support results."
    )

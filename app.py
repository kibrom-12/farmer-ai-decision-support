import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from catboost import CatBoostRegressor, CatBoostClassifier

# ============================================================
# APP SETUP
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
# CROP pH RANGES
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
    "55.0": (6.0, 7.0),
    "56.0": (6.0, 7.5),
    "61.0": (6.0, 7.5),

    "42.0": (5.5, 6.5),
    "46.0": (5.5, 7.5),
    "47.0": (5.5, 6.5),
    "48.0": (6.0, 6.5),
    "84.0": (5.5, 6.5),

    "71.0": (6.0, 7.0),

    "72.0": (5.0, 6.0),
    "74.0": (5.5, 6.5),
    "75.0": (6.0, 7.0),
    "76.0": (6.0, 7.5),

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
# pH ADJUSTMENT
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

    if ph_val < min_ph:

        difference = min_ph - ph_val

        return max(
            0.1,
            1.0 - (
                difference * 0.35
            )
        )

    difference = ph_val - max_ph

    return max(
        0.1,
        1.0 - (
            difference * 0.35
        )
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

    row["s4q01b"] = str(
        crop_code
    )

    row["s3q08"] = (
        float(area_ha)
        * 10000.0
    )

    row["lat_mod"] = float(
        latitude
    )

    row["lon_mod"] = float(
        longitude
    )

    for feature in CATEGORICAL_FEATURES:

        if feature == "s4q01b":

            row[feature] = str(
                crop_code
            )

        else:

            row[feature] = "0"

    df = pd.DataFrame(
        [row],
        columns=REC_FEATURES
    )

    for column in CATEGORICAL_FEATURES:

        df[column] = (
            df[column]
            .astype(str)
        )

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

        predicted_log_yield = float(
            yield_model.predict(
                farm
            )[0]
        )

        base_yield = max(
            0.0,
            float(
                np.expm1(
                    predicted_log_yield
                )
            )
        )

        ph_multiplier = (
            calculate_ph_multiplier(
                code,
                ph_val
            )
        )

        adjusted_yield = (
            base_yield
            * ph_multiplier
        )

        results.append({

            "Crop": CROP_NAMES[code],

            "Crop Code": code,

            "Crop Group": CROP_GROUPS[code],

            "Predicted Yield (kg/ha)": (
                adjusted_yield
            )
        })

    df = pd.DataFrame(
        results
    )

    minimum = df[
        "Predicted Yield (kg/ha)"
    ].min()

    maximum = df[
        "Predicted Yield (kg/ha)"
    ].max()

    if maximum > minimum:

        df["Decision Degree (%)"] = (

            (
                (
                    df[
                        "Predicted Yield (kg/ha)"
                    ]
                    - minimum
                )
                /
                (
                    maximum
                    - minimum
                )
            )
            * 100

        ).clip(
            0,
            100
        ).round(1)

    else:

        df["Decision Degree (%)"] = 100.0

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
# RAINFALL FORECAST
# ============================================================

def forecast_rain(
    days=7
):

    history = load_rainfall().copy()

    (
        _,
        rain_event_model,
        rain_amount_model
    ) = load_models()

    history["time"] = pd.to_datetime(
        history["time"]
    )

    history = history.sort_values(
        "time"
    ).reset_index(
        drop=True
    )

    rain_values = (
        history[
            "rain_sum"
        ]
        .astype(float)
        .tolist()
    )

    temperature_values = (
        history[
            "temperature_2m_mean"
        ]
        .astype(float)
        .tolist()
    )

    last_date = history[
        "time"
    ].iloc[-1]

    positive_rain = [
        value
        for value in rain_values
        if value > 0
    ]

    if len(positive_rain) > 0:

        average_historical_rain = (
            float(
                np.mean(
                    positive_rain
                )
            )
        )

    else:

        average_historical_rain = 2.5

    forecasts = []

    for step in range(
        1,
        days + 1
    ):

        forecast_date = (
            last_date
            + pd.Timedelta(
                days=step
            )
        )

        def lag(
            values,
            number
        ):

            if len(values) >= number:

                return float(
                    values[-number]
                )

            return 0.0

        def rolling_mean(
            values,
            number
        ):

            if len(values) >= number:

                return float(
                    np.mean(
                        values[-number:]
                    )
                )

            return float(
                np.mean(values)
            )

        day_of_year = (
            forecast_date.dayofyear
        )

        x = pd.DataFrame(
            [{
                "rain_lag_1":
                    lag(
                        rain_values,
                        1
                    ),

                "rain_lag_2":
                    lag(
                        rain_values,
                        2
                    ),

                "rain_lag_3":
                    lag(
                        rain_values,
                        3
                    ),

                "rain_lag_7":
                    lag(
                        rain_values,
                        7
                    ),

                "rain_lag_14":
                    lag(
                        rain_values,
                        14
                    ),

                "rain_lag_21":
                    lag(
                        rain_values,
                        21
                    ),

                "rain_lag_28":
                    lag(
                        rain_values,
                        28
                    ),

                "rain_roll_3":
                    rolling_mean(
                        rain_values,
                        3
                    ),

                "rain_roll_7":
                    rolling_mean(
                        rain_values,
                        7
                    ),

                "rain_roll_14":
                    rolling_mean(
                        rain_values,
                        14
                    ),

                "rain_roll_28":
                    rolling_mean(
                        rain_values,
                        28
                    ),

                "temp_lag_1":
                    lag(
                        temperature_values,
                        1
                    ),

                "temp_lag_7":
                    lag(
                        temperature_values,
                        7
                    ),

                "temp_lag_14":
                    lag(
                        temperature_values,
                        14
                    ),

                "dayofyear":
                    day_of_year,

                "month":
                    forecast_date.month,

                "sin_doy":
                    np.sin(
                        2
                        * np.pi
                        * day_of_year
                        / 365.25
                    ),

                "cos_doy":
                    np.cos(
                        2
                        * np.pi
                        * day_of_year
                        / 365.25
                    )
            }],
            columns=RAIN_FEATURES
        )

        rain_probability = float(
            rain_event_model
            .predict_proba(
                x
            )[0, 1]
        )

        predicted_log_amount = float(
            rain_amount_model.predict(
                x
            )[0]
        )

        raw_rain = max(
            0.0,
            float(
                np.expm1(
                    predicted_log_amount
                )
            )
        )

        if rain_probability >= 0.30:

            predicted_rain = raw_rain

        else:

            predicted_rain = (
                average_historical_rain
                * rain_probability
            )

        rain_values.append(
            predicted_rain
        )

        temperature_values.append(
            lag(
                temperature_values,
                1
            )
        )

        forecasts.append({

            "Date":
                forecast_date.strftime(
                    "%Y-%m-%d"
                ),

            "Rain Probability (%)":
                round(
                    rain_probability * 100,
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
# RAINFALL RISK
# ============================================================

def calculate_rainfall_risk(
    rainfall_df
):

    total_rain = float(
        rainfall_df[
            "Predicted Rain (mm)"
        ].sum()
    )

    rainy_days = int(
        (
            rainfall_df[
                "Predicted Rain (mm)"
            ]
            > 1
        ).sum()
    )

    heavy_rain_days = int(
        (
            rainfall_df[
                "Predicted Rain (mm)"
            ]
            >= 10
        ).sum()
    )

    average_probability = float(
        rainfall_df[
            "Rain Probability (%)"
        ].mean()
    )

    if total_rain < 5:

        dryness_risk = 80

    elif total_rain < 15:

        dryness_risk = 50

    else:

        dryness_risk = 20

    if heavy_rain_days >= 2:

        excess_rain_risk = 80

    elif heavy_rain_days == 1:

        excess_rain_risk = 50

    else:

        excess_rain_risk = 20

    probability_risk = (
        100
        - average_probability
    )

    risk_score = (
        0.45 * dryness_risk
        + 0.35 * excess_rain_risk
        + 0.20 * probability_risk
    )

    if risk_score < 35:

        risk_level = "Low"

    elif risk_score < 55:

        risk_level = "Moderate"

    elif risk_score < 75:

        risk_level = "High"

    else:

        risk_level = "Very High"

    return (
        round(risk_score, 1),
        risk_level,
        total_rain,
        rainy_days,
        heavy_rain_days
    )


# ============================================================
# TEMPORARY pH GUIDANCE
# ============================================================

def fertilizer_advice(
    ph_val
):

    if ph_val < 5.5:

        return (
            "Soil acidity requires attention",
            "Strongly acidic soil. Soil testing and "
            "local agronomic advice should be used "
            "before selecting fertilizer."
        )

    elif ph_val < 6.5:

        return (
            "Nutrient management should be soil-test based",
            "Acidic soil. Actual N/P/K measurements "
            "are needed for a reliable fertilizer recommendation."
        )

    elif ph_val <= 7.0:

        return (
            "Balanced nutrient management may be suitable",
            "Near-neutral pH. The final fertilizer recommendation "
            "must also consider measured soil N, P and K."
        )

    elif ph_val <= 7.5:

        return (
            "Monitor nutrient availability",
            "Slightly alkaline soil. Soil nutrient measurements "
            "are required for fertilizer selection."
        )

    else:

        return (
            "Alkaline soil requires attention",
            "Use soil-test results and locally validated "
            "agronomic recommendations."
        )


# ============================================================
# PAGE TITLE
# ============================================================

st.title(
    "🌾 AI-Powered Farmer Decision Support System"
)

st.markdown(
    """
    An integrated decision-support system combining:

    **🌱 Crop Recommendation • 📈 Yield Prediction • 🌧️ Rainfall Forecasting •
    ⚠️ Agricultural Risk • 🧪 Soil NPK Information • 💊 Fertilizer Decision Support**
    """
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "👨‍🌾 Farmer & Farm Information"
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

st.sidebar.header(
    "🌱 Farm Information"
)

area_ha = st.sidebar.number_input(
    "Farm Area (Hectares)",
    min_value=0.1,
    max_value=100.0,
    value=1.0,
    step=0.1
)

lat = st.sidebar.number_input(
    "Latitude",
    value=8.54
)

lon = st.sidebar.number_input(
    "Longitude",
    value=38.98
)

st.sidebar.header(
    "🧪 Soil Information"
)

ph = st.sidebar.slider(
    "Soil pH",
    min_value=4.0,
    max_value=9.0,
    value=6.5,
    step=0.1
)

# ============================================================
# CEREAL FILTER
# ============================================================

focus_cereals = st.sidebar.checkbox(
    "🌾 Focus on cereal crops",
    value=False,
    help=(
        "When selected, the recommendation is limited "
        "to cereal crops such as Sorghum, Maize, Wheat, "
        "Barley and Teff."
    )
)

# ============================================================
# NPK INPUT
# ============================================================

st.sidebar.subheader(
    "🧪 Soil NPK Measurements"
)

soil_n = st.sidebar.number_input(
    "Nitrogen (N)",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help=(
        "Enter the measured soil nitrogen value "
        "from laboratory analysis or a calibrated sensor."
    )
)

soil_p = st.sidebar.number_input(
    "Phosphorus (P)",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help=(
        "Enter the measured soil phosphorus value "
        "from laboratory analysis or a calibrated sensor."
    )
)

soil_k = st.sidebar.number_input(
    "Potassium (K)",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help=(
        "Enter the measured soil potassium value "
        "from laboratory analysis or a calibrated sensor."
    )
)

st.sidebar.caption(
    "N/P/K values are inputs only until the validated "
    "fertilizer AI model is connected."
)

# ============================================================
# RUN BUTTON
# ============================================================

run_analysis = st.sidebar.button(
    "🚀 Run AI Analysis",
    type="primary",
    use_container_width=True
)

# ============================================================
# MAIN ANALYSIS
# ============================================================

if run_analysis:

    with st.spinner(
        "Running crop, yield and rainfall AI models..."
    ):

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

        (
            risk_score,
            risk_level,
            total_rain,
            rainy_days,
            heavy_rain_days
        ) = calculate_rainfall_risk(
            rain_df
        )

    # ========================================================
    # LOCATION
    # ========================================================

    st.info(
        f"""
        **📍 Location Selected:** {woreda}, {zone}, {region}

        **Coordinates:** {lat:.4f}° N, {lon:.4f}° E
        """
    )

    # ========================================================
    # TOP SUMMARY METRICS
    # ========================================================

    st.header(
        "📊 AI Decision Summary"
    )

    summary1, summary2, summary3, summary4 = st.columns(4)

    with summary1:

        st.metric(
            "Recommended Crop",
            top_crop["Crop"]
        )

    with summary2:

        st.metric(
            "Predicted Yield",
            f"{top_crop['Predicted Yield (kg/ha)']:,.1f} kg/ha"
        )

    with summary3:

        st.metric(
            "Decision Degree",
            f"{top_crop['Decision Degree (%)']:.1f}%"
        )

    with summary4:

        st.metric(
            "Rainfall Risk",
            risk_level
        )

    # ========================================================
    # RECOMMENDED CROP
    # ========================================================

    st.header(
        "🏆 Recommended Crop"
    )

    st.success(
        f"""
        **{top_crop['Crop']}**

        Predicted yield:
        **{top_crop['Predicted Yield (kg/ha)']:,.2f} kg/ha**

        Decision degree:
        **{top_crop['Decision Degree (%)']:.1f}%**

        Crop group:
        **{top_crop['Crop Group']}**
        """
    )

    if focus_cereals:

        st.caption(
            "🌾 Cereal-focus mode is ON. "
            "Only cereal crops are being compared."
        )

    else:

        st.caption(
            "🌱 All available crop groups are being compared."
        )

    # ========================================================
    # CROP RANKING
    # ========================================================

    st.header(
        "🌱 Crop Recommendation Ranking"
    )

    top_10 = crop_df.head(10)

    fig_crops = px.bar(
        top_10,
        x="Crop",
        y="Predicted Yield (kg/ha)",
        color="Decision Degree (%)",
        text_auto=".1f",
        title="Top 10 Crop Yield Predictions",
        color_continuous_scale="Greens"
    )

    fig_crops.update_layout(
        xaxis_title="Crop",
        yaxis_title="Predicted Yield (kg/ha)"
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

    # ========================================================
    # RAINFALL
    # ========================================================

    st.header(
        "🌧️ AI 7-Day Rainfall Forecast"
    )

    rain1, rain2, rain3, rain4 = st.columns(4)

    with rain1:

        st.metric(
            "7-Day Rainfall",
            f"{total_rain:.2f} mm"
        )

    with rain2:

        st.metric(
            "Rainy Days",
            rainy_days
        )

    with rain3:

        st.metric(
            "Heavy Rain Days",
            heavy_rain_days
        )

    with rain4:

        st.metric(
            "Risk",
            f"{risk_level}"
        )

    chart_col, table_col = st.columns(
        [2, 1]
    )

    with chart_col:

        fig_rain = go.Figure()

        fig_rain.add_trace(
            go.Bar(
                x=rain_df["Date"],
                y=rain_df[
                    "Predicted Rain (mm)"
                ],
                name="Predicted Rain (mm)",
                marker_color="#1f77b4"
            )
        )

        fig_rain.add_trace(
            go.Scatter(
                x=rain_df["Date"],
                y=rain_df[
                    "Rain Probability (%)"
                ],
                name="Rain Probability (%)",
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
                "and Rain Probability"
            ),
            xaxis_title="Date",
            yaxis=dict(
                title="Predicted Rain (mm)"
            ),
            yaxis2=dict(
                title="Rain Probability (%)",
                overlaying="y",
                side="right",
                range=[0, 100]
            ),
            legend=dict(
                x=0.01,
                y=0.99
            )
        )

        st.plotly_chart(
            fig_rain,
            use_container_width=True
        )

    with table_col:

        st.subheader(
            "📋 Daily Forecast"
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

    # ========================================================
    # RAINFALL RISK
    # ========================================================

    st.header(
        "⚠️ Agricultural Rainfall Risk"
    )

    if risk_level == "Low":

        st.success(
            f"Rainfall risk is **Low** ({risk_score:.1f}/100)."
        )

    elif risk_level == "Moderate":

        st.info(
            f"Rainfall risk is **Moderate** ({risk_score:.1f}/100)."
        )

    elif risk_level == "High":

        st.warning(
            f"Rainfall risk is **High** ({risk_score:.1f}/100)."
        )

    else:

        st.error(
            f"Rainfall risk is **Very High** ({risk_score:.1f}/100)."
        )

    st.write(
        f"""
        **Risk score:** {risk_score:.1f}/100

        **Expected rainfall:** {total_rain:.2f} mm

        **Rainy days:** {rainy_days}

        **Heavy-rain days:** {heavy_rain_days}
        """
    )

    # ========================================================
    # SOIL NPK
    # ========================================================

    st.header(
        "🧪 Soil Nutrient Diagnosis"
    )

    npk1, npk2, npk3, npk4 = st.columns(4)

    with npk1:

        st.metric(
            "Soil pH",
            f"{ph:.1f}"
        )

    with npk2:

        st.metric(
            "Nitrogen (N)",
            f"{soil_n:.2f}"
        )

    with npk3:

        st.metric(
            "Phosphorus (P)",
            f"{soil_p:.2f}"
        )

    with npk4:

        st.metric(
            "Potassium (K)",
            f"{soil_k:.2f}"
        )

    st.caption(
        "N/P/K values shown here are the measurements "
        "entered by the farmer. Their units and interpretation "
        "must match the laboratory or sensor source."
    )

    # ========================================================
    # FERTILIZER AI
    # ========================================================

    st.header(
        "💊 AI Fertilizer Decision Support"
    )

    st.info(
        """
        **N/P/K fertilizer AI status: Preparing validated model**

        The system now accepts soil N, P, K and pH measurements.

        The final fertilizer recommendation will be connected
        after the Ethiopian soil dataset and appropriate
        agronomic fertilizer-response/reference labels are
        validated.

        This prevents the system from inventing fertilizer
        recommendations from unsupported thresholds.
        """
    )

    # ========================================================
    # TEMPORARY pH GUIDANCE
    # ========================================================

    st.subheader(
        "📌 Current pH-Based Guidance"
    )

    fertilizer_name, fertilizer_description = (
        fertilizer_advice(ph)
    )

    st.warning(
        f"""
        **Guidance:** {fertilizer_name}

        {fertilizer_description}

        ⚠️ This is NOT the final N/P/K fertilizer AI.
        Soil pH alone cannot determine actual N, P or K deficiency.
        """
    )

    # ========================================================
    # SOIL DECISION STATUS
    # ========================================================

    st.subheader(
        "🔬 Soil Decision Status"
    )

    if (
        soil_n == 0
        and soil_p == 0
        and soil_k == 0
    ):

        st.warning(
            "N/P/K measurements have not been entered yet."
        )

    else:

        st.success(
            "N/P/K measurements are available for this farm scenario."
        )

    # ========================================================
    # WHAT-IF ANALYSIS
    # ========================================================

    st.header(
        "🔄 What-If Analysis"
    )

    st.write(
        "Change soil pH to see how the existing crop-yield "
        "recommendation changes."
    )

    what_if_ph = st.slider(
        "What-if Soil pH",
        min_value=4.0,
        max_value=9.0,
        value=float(ph),
        step=0.1,
        key="what_if_ph"
    )

    what_if_crops = predict_crops(
        area_ha,
        lat,
        lon,
        what_if_ph
    )

    if focus_cereals:

        what_if_crops = what_if_crops[
            what_if_crops[
                "Crop Group"
            ] == "Cereal"
        ].copy()

        what_if_crops = (
            what_if_crops
            .sort_values(
                "Decision Degree (%)",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        what_if_crops["Rank"] = (
            what_if_crops.index + 1
        )

    what_if_top = (
        what_if_crops.iloc[0]
    )

    st.success(
        f"""
        At soil pH **{what_if_ph:.1f}**,

        the model's top crop is
        **{what_if_top['Crop']}**

        with predicted yield of
        **{what_if_top['Predicted Yield (kg/ha)']:,.2f} kg/ha**.
        """
    )

    # ========================================================
    # MODEL EXPLAINABILITY
    # ========================================================

    st.header(
        "🔍 Explainable AI"
    )

    st.write(
        """
        The yield model is a CatBoost model trained using
        agricultural, environmental, soil-related and
        geographic features.

        The table below shows the model's global feature
        importance for the trained yield model.
        """
    )

    try:

        yield_model, _, _ = load_models()

        importance_values = (
            yield_model
            .get_feature_importance()
        )

        importance_df = pd.DataFrame({

            "Feature":
                REC_FEATURES,

            "Importance":
                importance_values

        }).sort_values(
            "Importance",
            ascending=False
        ).head(15)

        fig_importance = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=(
                "Top 15 Yield Model Features"
            )
        )

        fig_importance.update_layout(
            yaxis={
                "categoryorder":
                "total ascending"
            }
        )

        st.plotly_chart(
            fig_importance,
            use_container_width=True
        )

        st.caption(
            "Feature importance shows model influence, "
            "not causation."
        )

    except Exception as error:

        st.warning(
            f"Feature importance is temporarily unavailable: {error}"
        )

    # ========================================================
    # FARM SUMMARY
    # ========================================================

    st.header(
        "📋 Farmer Decision Summary"
    )

    summary_col1, summary_col2 = st.columns(
        2
    )

    with summary_col1:

        st.subheader(
            "👨‍🌾 Farm"
        )

        st.write(
            f"**Region:** {region}"
        )

        st.write(
            f"**Zone:** {zone}"
        )

        st.write(
            f"**Woreda:** {woreda}"
        )

        st.write(
            f"**Area:** {area_ha:.2f} ha"
        )

        st.write(
            f"**Latitude:** {lat:.4f}"
        )

        st.write(
            f"**Longitude:** {lon:.4f}"
        )

    with summary_col2:

        st.subheader(
            "🧪 Soil"
        )

        st.write(
            f"**pH:** {ph:.1f}"
        )

        st.write(
            f"**N:** {soil_n:.2f}"
        )

        st.write(
            f"**P:** {soil_p:.2f}"
        )

        st.write(
            f"**K:** {soil_k:.2f}"
        )

        st.write(
            f"**Cereal focus:** "
            f"{'Yes' if focus_cereals else 'No'}"
        )

    # ========================================================
    # FUTURE IoT
    # ========================================================

    st.header(
        "🔌 Future IoT Soil Sensor Integration"
    )

    st.info(
        """
        **Planned architecture:**

        🧪 NPK / pH / moisture / temperature sensors
        ↓
        📡 ESP32 or similar edge device
        ↓
        🌐 Real-time soil measurements
        ↓
        🤖 Soil nutrient diagnosis
        ↓
        💊 Fertilizer AI
        ↓
        🌾 Crop + Yield + Weather Decision Support

        The future sensor system will use the same AI
        decision-support architecture after sensor calibration
        and validation against appropriate soil laboratory data.
        """
    )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.header(
        "⚠️ Decision-Support Disclaimer"
    )

    st.warning(
        """
        This system provides agricultural decision support,
        not guaranteed agronomic outcomes.

        Crop and yield predictions depend on the training data
        and farmer inputs.

        The current fertilizer component is not yet a validated
        N/P/K dosage model. Exact fertilizer rates should only
        be provided when supported by appropriate fertilizer
        response/reference data and local agronomic validation.

        Soil sensor measurements should be calibrated and,
        where possible, compared with laboratory soil analysis.
        """
    )

else:

    # ========================================================
    # INITIAL LANDING SCREEN
    # ========================================================

    st.info(
        "👈 Enter the farmer and soil information on the left, "
        "then click **🚀 Run AI Analysis**."
    )

    st.header(
        "🌾 System Workflow"
    )

    workflow1, workflow2, workflow3, workflow4 = st.columns(4)

    with workflow1:

        st.subheader(
            "1️⃣ Farm"
        )

        st.write(
            "Location, farm area and soil information."
        )

    with workflow2:

        st.subheader(
            "2️⃣ Soil"
        )

        st.write(
            "pH and measured N/P/K inputs."
        )

    with workflow3:

        st.subheader(
            "3️⃣ AI"
        )

        st.write(
            "Crop, yield and rainfall models."
        )

    with workflow4:

        st.subheader(
            "4️⃣ Decision"
        )

        st.write(
            "Crop ranking, rainfall risk and fertilizer support."
        )

    st.markdown(
        "---"
    )

    st.header(
        "🧪 Soil NPK Component"
    )

    st.write(
        """
        The application is now prepared to accept real soil
        nitrogen, phosphorus and potassium measurements.

        Once the Ethiopian soil dataset is approved and obtained,
        these inputs will be connected to the validated fertilizer
        recommendation model.
        """
    )

    st.header(
        "🔌 Future IoT"
    )

    st.write(
        """
        The same N/P/K interface can later receive measurements
        automatically from calibrated IoT soil sensors through
        an ESP32 or similar device.
        """
    )

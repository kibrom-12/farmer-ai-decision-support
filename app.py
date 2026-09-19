# ============================================================
# NPK / FERTILIZER RECOMMENDATION
# ============================================================

st.header("🧪 AI Fertilizer Recommendation")

st.write(
    "The system combines the entered soil measurements with "
    "Ethiopian fertilizer-response trial data."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Soil pH",
        f"{ph:.1f}"
    )

with col2:
    st.metric(
        "Nitrogen (N)",
        f"{soil_n:.2f}%"
    )

with col3:
    st.metric(
        "Phosphorus (P)",
        f"{soil_p:.1f} ppm"
    )

try:

    fertilizer_data = load_fertilizer_data()

    fertilizer_result = recommend_fertilizer(
        soil_pH=ph,
        soil_N_pct=soil_n,
        soil_P_ppm=soil_p,
        modeling_table=fertilizer_data
    )

    # --------------------------------------------------------
    # MAIN RECOMMENDATION
    # --------------------------------------------------------

    st.subheader("🌱 Recommended Fertilizer Strategy")

    rec_col, conf_col = st.columns(2)

    with rec_col:

        st.success(
            f"### {fertilizer_result['recommended_strategy']}"
        )

    with conf_col:

        st.info(
            f"### Confidence: "
            f"{fertilizer_result['confidence']}"
        )

    # --------------------------------------------------------
    # ESTIMATED YIELD
    # --------------------------------------------------------

    st.metric(
        "Estimated Yield Under Recommended Strategy",
        f"{fertilizer_result['estimated_yield_kg_ha']:,.0f} kg/ha"
    )

    # --------------------------------------------------------
    # STRATEGY COMPARISON
    # --------------------------------------------------------

    st.subheader(
        "📊 Fertilizer Strategy Comparison"
    )

    ranked = fertilizer_result[
        "ranked_strategies"
    ]

    strategy_values = fertilizer_result[
        "estimated_yield_by_strategy"
    ]

    comparison = pd.DataFrame({
        "Fertilizer Strategy": ranked,
        "Estimated Yield (kg/ha)": [
            strategy_values[x]
            for x in ranked
        ]
    })

    comparison[
        "Estimated Yield (kg/ha)"
    ] = comparison[
        "Estimated Yield (kg/ha)"
    ].round(0)

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True
    )

    # --------------------------------------------------------
    # SOIL DIAGNOSIS
    # --------------------------------------------------------

    st.subheader(
        "🔬 Soil Diagnosis"
    )

    for diagnosis in fertilizer_result[
        "soil_diagnosis"
    ]:

        st.write(
            f"• {diagnosis}"
        )

    # --------------------------------------------------------
    # WHY
    # --------------------------------------------------------

    st.subheader(
        "💡 Why was this fertilizer strategy recommended?"
    )

    st.write(
        fertilizer_result["reason"]
    )

    # --------------------------------------------------------
    # IMPORTANT SCIENTIFIC WARNING
    # --------------------------------------------------------

    st.warning(
        fertilizer_result["warning"]
    )

except Exception as e:

    st.error(
        f"Fertilizer recommendation could not be generated: {e}"
    )

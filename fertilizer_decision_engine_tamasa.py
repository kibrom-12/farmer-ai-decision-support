from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

SOIL_FEATURES = ["pH           (1:2.5 water)", "TN                (%)", "Av. P (ppm)"]
STRATEGIES = ["NE Recommendation", "Regional Recommendation", "Soil Test Based Recommendation"]
YIELD_COLUMNS = {
    "Control": "Control_yield_kg_ha",
    "NE Recommendation": "NE Recommendation_yield_kg_ha",
    "Regional Recommendation": "Regional Recommendation_yield_kg_ha",
    "Soil Test Based Recommendation": "Soil Test Based Recommendation_yield_kg_ha",
}
GAIN_COLUMNS = {s: f"{s}_gain_vs_control_kg_ha" for s in STRATEGIES}
RATE_COLUMNS = {
    "NE Recommendation": ("NE Recomm._N", "NE Recomm._P", "NE Recomm._K"),
    "Regional Recommendation": ("Regional Recom._N", "Regional Recom._P", "Regional Recom._K"),
    "Soil Test Based Recommendation": ("Soil Test Based Recom._N", "Soil Test Based Recom._P", "Soil Test Based Recom._K"),
}


def load_tamasa_table(app_dir=None):
    app_dir = Path(app_dir) if app_dir else Path(__file__).parent
    path = app_dir / "TAMASA_clean_trial_modeling_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"TAMASA dataset not found: {path}")
    df = pd.read_csv(path)
    required = SOIL_FEATURES + list(YIELD_COLUMNS.values())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("TAMASA dataset missing columns: " + ", ".join(missing))
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def diagnose_soil(pH, N, P):
    out = []
    out.append("pH is relatively low." if pH < 5.5 else "pH is moderately acidic." if pH < 6 else "pH is broadly suitable in the TAMASA reference range." if pH <= 7.2 else "pH is relatively high compared with the TAMASA reference data.")
    out.append("Total N is relatively low." if N < 0.10 else "Total N is intermediate." if N < 0.20 else "Total N is relatively high compared with many TAMASA observations.")
    out.append("Available P is relatively low." if P < 10 else "Available P is intermediate." if P < 20 else "Available P is relatively high compared with many TAMASA observations.")
    out.append("Measured soil K is not used for matching because complete soil-K observations are unavailable in this TAMASA table.")
    return out
def farmer_fertilizer_recommendation(soil_N_pct, soil_P_ppm):
    """
    Converts TAMASA soil diagnosis into a farmer-facing
    nutrient recommendation.
    """

    n_low = soil_N_pct < 0.10
    p_low = soil_P_ppm < 10

    if n_low and p_low:
        return (
            "N + P fertilizer",
            "Both soil nitrogen and available phosphorus are relatively low."
        )

    if p_low:
        return (
            "P-containing fertilizer",
            "Available phosphorus is relatively low, so phosphorus should be included in the fertilizer strategy."
        )

    if n_low:
        return (
            "N-containing fertilizer",
            "Total soil nitrogen is relatively low, so nitrogen should be included in the fertilizer strategy."
        )

    return (
        "TAMASA-selected fertilizer strategy",
        "Soil nitrogen and available phosphorus are not classified as low under the current TAMASA thresholds."
    )

def _predict_gain(df, strategy, x):
    cols = SOIL_FEATURES + [GAIN_COLUMNS[strategy]]
    d = df[cols].dropna()
    if len(d) < 3:
        return np.nan
    scaler = StandardScaler()
    X = scaler.fit_transform(d[SOIL_FEATURES])
    X0 = scaler.transform(pd.DataFrame([x], columns=SOIL_FEATURES))
    k = min(7, len(d))
    model = KNeighborsRegressor(n_neighbors=k, weights="distance")
    model.fit(X, d[GAIN_COLUMNS[strategy]].to_numpy())
    return float(model.predict(X0)[0])


def _nearest_rates(df, strategy, x):
    cols = RATE_COLUMNS[strategy]
    d = df[SOIL_FEATURES + list(cols)].dropna(subset=SOIL_FEATURES)
    if d.empty:
        return {"N": np.nan, "P": np.nan, "K": np.nan}
    scaler = StandardScaler()
    X = scaler.fit_transform(d[SOIL_FEATURES])
    x0 = scaler.transform(pd.DataFrame([x], columns=SOIL_FEATURES))[0]
    dist = np.linalg.norm(X - x0, axis=1)
    idx = np.argsort(dist)[:min(7, len(d))]
    w = 1 / (dist[idx] + 1e-6)
    result = {}
    for nutrient, col in zip(["N", "P", "K"], cols):
        v = pd.to_numeric(d.iloc[idx][col], errors="coerce").to_numpy()
        mask = np.isfinite(v)
        result[nutrient] = float(np.average(v[mask], weights=w[mask])) if mask.any() else np.nan
    return result


def recommend_fertilizer(soil_pH, soil_N_pct, soil_P_ppm, modeling_table):
    x = {SOIL_FEATURES[0]: float(soil_pH), SOIL_FEATURES[1]: float(soil_N_pct), SOIL_FEATURES[2]: float(soil_P_ppm)}
    control = pd.to_numeric(modeling_table[YIELD_COLUMNS["Control"]], errors="coerce").mean()
    rows = []
    for strategy in STRATEGIES:
        gain = _predict_gain(modeling_table, strategy, x)
        rates = _nearest_rates(modeling_table, strategy, x)
        rows.append({
            "Fertilizer Strategy": strategy,
            "Predicted Gain vs Control (kg/ha)": gain,
            "Predicted Yield (kg/ha)": control + gain if np.isfinite(gain) else np.nan,
            "N Rate (kg/ha)": rates["N"], "P Rate (kg/ha)": rates["P"], "K Rate (kg/ha)": rates["K"],
        })
    ranking = pd.DataFrame(rows).sort_values("Predicted Gain vs Control (kg/ha)", ascending=False, na_position="last").reset_index(drop=True)
    recommendation = str(ranking.iloc[0]["Fertilizer Strategy"])
      
    return {
        "recommendation": recommendation,
        "confidence": "Low — experimental TAMASA baseline",
        "diagnosis": "\n".join("• " + s for s in diagnose_soil(soil_pH, soil_N_pct, soil_P_ppm)),
        "ranking": ranking,
        "validation_note": "TAMASA baseline: 50.0% raw strategy-selection accuracy and 30.8% balanced accuracy on the 24 complete pH/N/P trials under leave-one-out validation. This is decision support, not a guaranteed prescription.",
    }

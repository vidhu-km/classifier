import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("Viewfield Bakken Well Classification Tool")

# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------
FILE_PATH = "tables.xlsx"

try:
    sheet_names = pd.ExcelFile(FILE_PATH).sheet_names
    st.caption(f"Sheets found: {sheet_names}")

    field_eur = pd.read_excel(FILE_PATH, sheet_name=0)
    field_ip90 = pd.read_excel(FILE_PATH, sheet_name=1)
    field_1y = pd.read_excel(FILE_PATH, sheet_name=2)
    prospects = pd.read_excel(FILE_PATH, sheet_name=3)
except Exception as e:
    st.error(f"Could not load tables.xlsx: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# 2. COLUMN DETECTION
# ---------------------------------------------------------------------------
def find_col(df, keywords, exclude=None):
    for col in df.columns:
        low = col.lower()
        if exclude and any(ex.lower() in low for ex in exclude):
            continue
        if any(kw.lower() in low for kw in keywords):
            return col
    return None

# Field sheets
sec_col_eur = find_col(field_eur, ["section"], exclude=["ooip", "eur", "ip", "cuml"])
ooip_col = find_col(field_eur, ["ooip"])
eur_col = find_col(field_eur, ["eur"])
sec_col_ip90 = find_col(field_ip90, ["section"], exclude=["ip90"])
ip90_col = find_col(field_ip90, ["ip90"])
sec_col_1y = find_col(field_1y, ["section"], exclude=["1y", "cuml"])
y1_col = find_col(field_1y, ["1y", "cuml"])

# Prospect sheet
p_ooip = find_col(prospects, ["ooip"])
p_eur = find_col(prospects, ["eur"])
p_ip90 = find_col(prospects, ["ip90"])
p_1y = find_col(prospects, ["1y", "cuml"])
p_uwi = find_col(prospects, ["uwi", "well", "name"])

required = {
    "Section": sec_col_eur, "OOIP": ooip_col, "EUR": eur_col,
    "IP90": ip90_col, "1Y": y1_col,
    "Prospect OOIP": p_ooip, "Prospect EUR": p_eur, "Prospect IP90": p_ip90, "Prospect 1Y": p_1y
}
missing = [k for k, v in required.items() if v is None]
if missing:
    st.error(f"Could not detect columns: {missing}")
    st.stop()

# ---------------------------------------------------------------------------
# 3. MERGE FIELD DATA
# ---------------------------------------------------------------------------
field = field_eur[[sec_col_eur, ooip_col, eur_col]].copy()

field = field.merge(
    field_ip90[[sec_col_ip90, ip90_col]].rename(columns={sec_col_ip90: sec_col_eur}),
    on=sec_col_eur,
)

field = field.merge(
    field_1y[[sec_col_1y, y1_col]].rename(columns={sec_col_1y: sec_col_eur}),
    on=sec_col_eur,
)

field.columns = ["Section", "OOIP", "EUR", "IP90", "1Y"]
field.dropna(inplace=True)

# ---------------------------------------------------------------------------
# 4. FIT FIELD TRENDS
# ---------------------------------------------------------------------------
def fit_trend(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    model = LinearRegression().fit(
        x[mask].values.reshape(-1, 1),
        y[mask].values
    )
    return model

eur_model = fit_trend(field["OOIP"], field["EUR"])
ip90_model = fit_trend(field["OOIP"], field["IP90"])
y1_model = fit_trend(field["OOIP"], field["1Y"])

# Field residuals
field["EUR_resid"] = field["EUR"] - eur_model.predict(field["OOIP"].values.reshape(-1, 1))
field["IP90_resid"] = field["IP90"] - ip90_model.predict(field["OOIP"].values.reshape(-1, 1))
field["Y1_resid"] = field["1Y"] - y1_model.predict(field["OOIP"].values.reshape(-1, 1))

eur_std = field["EUR_resid"].std()
ip90_std = field["IP90_resid"].std()
y1_std = field["Y1_resid"].std()

# ---------------------------------------------------------------------------
# 5. SCORE PROSPECTS
# ---------------------------------------------------------------------------
pros = prospects.rename(columns={
    p_ooip: "SectionOOIP",
    p_eur: "Projected EUR",
    p_ip90: "Projected IP90",
    p_1y: "Projected 1Y"
})

if p_uwi:
    pros = pros.rename(columns={p_uwi: "UWI"})
else:
    pros["UWI"] = pros.index.astype(str)

pros.dropna(subset=["SectionOOIP", "Projected EUR", "Projected IP90", "Projected 1Y"], inplace=True)

# Predictions
pros["EUR_pred"] = eur_model.predict(pros["SectionOOIP"].values.reshape(-1, 1))
pros["IP90_pred"] = ip90_model.predict(pros["SectionOOIP"].values.reshape(-1, 1))
pros["Y1_pred"] = y1_model.predict(pros["SectionOOIP"].values.reshape(-1, 1))

# Residuals
pros["EUR_resid"] = pros["Projected EUR"] - pros["EUR_pred"]
pros["IP90_resid"] = pros["Projected IP90"] - pros["IP90_pred"]
pros["Y1_resid"] = pros["Projected 1Y"] - pros["Y1_pred"]

# Z-scores
pros["Z_EUR"] = pros["EUR_resid"] / eur_std
pros["Z_IP90"] = pros["IP90_resid"] / ip90_std
pros["Z_1Y"] = pros["Y1_resid"] / y1_std

# ---------------------------------------------------------------------------
# 6. SIDEBAR SETTINGS: user must provide 3 weights summing to 100
# ---------------------------------------------------------------------------
st.sidebar.header("Classification Settings (weights must sum to 100)")

w_eur = st.sidebar.number_input("Weight EUR (%)", min_value=0, max_value=100, value=50, step=1)
w_y1 = st.sidebar.number_input("Weight 1Y (%)", min_value=0, max_value=100, value=25, step=1)
w_ip90 = st.sidebar.number_input("Weight IP90 (%)", min_value=0, max_value=100, value=25, step=1)

weight_sum = w_eur + w_y1 + w_ip90

if weight_sum != 100:
    st.sidebar.error(f"Weights must sum to 100! Current sum: {weight_sum}")
    st.stop()

# Convert percentages to fractions for calculation
w_eur /= 100
w_y1 /= 100
w_ip90 /= 100

threshold = st.sidebar.slider(
    "Composite Z-score threshold (σ)",
    0.1, 2.0, 0.5, 0.05
)

# ---------------------------------------------------------------------------
# Weighted composite Z-score
# ---------------------------------------------------------------------------
pros["Composite_Z"] = (
    w_eur * pros["Z_EUR"] +
    w_y1 * pros["Z_1Y"] +
    w_ip90 * pros["Z_IP90"]
)

def classify(z):
    if z > threshold:
        return "Above Trend"
    elif z < -threshold:
        return "Below Trend"
    else:
        return "On Trend"

pros["Classification"] = pros["Composite_Z"].apply(classify)

# ---------------------------------------------------------------------------
# 7. PLOTS
# ---------------------------------------------------------------------------
color_map = {
    "Above Trend": "#2ca02c",
    "On Trend": "#1f77b4",
    "Below Trend": "#d62728"
}

st.markdown("## 📊 Performance Charts")

col1, col2 = st.columns(2)

with col1:
    fig_eur = px.scatter(
        pros, x="SectionOOIP", y="Projected EUR",
        color="Classification", color_discrete_map=color_map,
        hover_data=["UWI"], title="EUR vs Section OOIP"
    )
    fig_eur.add_trace(go.Scatter(
        x=field["OOIP"], y=field["EUR"],
        mode="markers", name="Field Wells",
        marker=dict(color="lightgrey", size=6)
    ))
    st.plotly_chart(fig_eur, use_container_width=True)

    fig_ip90 = px.scatter(
        pros, x="SectionOOIP", y="Projected IP90",
        color="Classification", color_discrete_map=color_map,
        hover_data=["UWI"], title="IP90 vs Section OOIP"
    )
    fig_ip90.add_trace(go.Scatter(
        x=field["OOIP"], y=field["IP90"],
        mode="markers", name="Field Wells",
        marker=dict(color="lightgrey", size=6)
    ))
    st.plotly_chart(fig_ip90, use_container_width=True)

with col2:
    fig_y1 = px.scatter(
        pros, x="SectionOOIP", y="Projected 1Y",
        color="Classification", color_discrete_map=color_map,
        hover_data=["UWI"], title="1Y Cumulative vs Section OOIP"
    )
    fig_y1.add_trace(go.Scatter(
        x=field["OOIP"], y=field["1Y"],
        mode="markers", name="Field Wells",
        marker=dict(color="lightgrey", size=6)
    ))
    st.plotly_chart(fig_y1, use_container_width=True)

    fig_comp = px.scatter(
        pros, x="SectionOOIP", y="Composite_Z",
        color="Classification", color_discrete_map=color_map,
        hover_data=["UWI"], title="Weighted Composite Z-Score"
    )
    fig_comp.add_hline(y=threshold, line_dash="dot", line_color="green")
    fig_comp.add_hline(y=-threshold, line_dash="dot", line_color="red")
    fig_comp.add_hline(y=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig_comp, use_container_width=True)

# ---------------------------------------------------------------------------
# 8. SUMMARY TABLE
# ---------------------------------------------------------------------------
st.markdown("## Summary")
summary = pros["Classification"].value_counts().reset_index()
summary.columns = ["Classification", "Count"]
st.dataframe(summary, use_container_width=True)

# ---------------------------------------------------------------------------
# 9. DETAILED TABLE + DOWNLOAD
# ---------------------------------------------------------------------------
st.markdown("## Classified Prospects")
display_cols = [
    "UWI", "SectionOOIP",
    "Projected EUR", "Projected 1Y", "Projected IP90",
    "Z_EUR", "Z_1Y", "Z_IP90",
    "Composite_Z", "Classification"
]
st.dataframe(pros[display_cols], use_container_width=True)

st.download_button(
    label="📥 Download Classified Prospects",
    data=pros[display_cols].to_csv(index=False),
    file_name="classified_prospects.csv",
    mime="text/csv",
)
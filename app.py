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
# 6. SIDEBAR SETTINGS
# ---------------------------------------------------------------------------
st.sidebar.header("Classification Settings")

threshold = st.sidebar.slider(
    "Composite Z-score threshold (σ)",
    0.1, 2.0, 0.5, 0.1
)

st.sidebar.markdown(
    """
    Weighted composite classification:

    - EUR: 50%  
    - 1Y cumulative: 25%  
    - IP90: 25%  

    The slider controls how many composite σ deviations classify Above/Below Trend.
    """
)

# Weighted composite Z-score
pros["Composite_Z"] = 0.5 * pros["Z_EUR"] + 0.25 * pros["Z_1Y"] + 0.25 * pros["Z_IP90"]

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
st.markdown("## 📊 Prospect vs Field Trend (Weighted Composite Classification)")

color_map = {
    "Above Trend": "#2ca02c",
    "On Trend": "#1f77b4",
    "Below Trend": "#d62728"
}

col1, col2 = st.columns(2)

with col1:
    fig1 = px.scatter(
        pros,
        x="SectionOOIP",
        y="Projected EUR",
        color="Classification",
        color_discrete_map=color_map,
        hover_data=["UWI"],
        title="Projected EUR vs Section OOIP"
    )

    fig1.add_trace(go.Scatter(
        x=field["OOIP"],
        y=field["EUR"],
        mode="markers",
        name="Field Wells",
        marker=dict(color="lightgrey", size=6)
    ))

    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.scatter(
        pros,
        x="SectionOOIP",
        y="Composite_Z",
        color="Classification",
        color_discrete_map=color_map,
        hover_data=["UWI"],
        title="Weighted Composite Z-Score"
    )

    fig2.add_hline(y=threshold, line_dash="dot", line_color="green")
    fig2.add_hline(y=-threshold, line_dash="dot", line_color="red")
    fig2.add_hline(y=0, line_dash="dash", line_color="black")

    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# 8. TABLES
# ---------------------------------------------------------------------------
st.markdown("## Classified Prospects")

display_cols = [
    "UWI", "SectionOOIP",
    "Projected EUR", "Projected 1Y", "Projected IP90",
    "Z_EUR", "Z_1Y", "Z_IP90",
    "Composite_Z", "Classification"
]

st.dataframe(pros[display_cols], use_container_width=True)

# ---------------------------------------------------------------------------
# 9. DOWNLOAD
# ---------------------------------------------------------------------------
st.download_button(
    label="📥 Download Classified Prospects",
    data=pros[display_cols].to_csv(index=False),
    file_name="classified_prospects.csv",
    mime="text/csv",
)
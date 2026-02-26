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
    """Return first column containing any keyword (case-insensitive)."""
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
sec_col_ip90 = find_col(field_ip90, ["section"], exclude=["ip90", "ip", "eur"])
ip90_col = find_col(field_ip90, ["ip90", "ip 90", "average section ip90"])
sec_col_1y = find_col(field_1y, ["section"], exclude=["cuml", "1y"])
y1_col = find_col(field_1y, ["1y", "cuml", "cumul"])

# Prospect sheet
p_ooip = find_col(prospects, ["ooip"])
p_eur = find_col(prospects, ["eur"])
p_ip90 = find_col(prospects, ["ip90"])
p_uwi = find_col(prospects, ["uwi", "well", "name"])

# Validate
required = {
    "Section (EUR)": sec_col_eur, "OOIP": ooip_col, "EUR": eur_col,
    "Section (IP90)": sec_col_ip90, "IP90": ip90_col,
    "Section (1Y)": sec_col_1y, "1Y Cuml": y1_col,
    "Prospect OOIP": p_ooip, "Prospect EUR": p_eur, "Prospect IP90": p_ip90,
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
field.dropna(subset=["OOIP", "EUR", "IP90", "1Y"], inplace=True)

# ---------------------------------------------------------------------------
# 4. FIT TRENDS FROM FIELD DATA
# ---------------------------------------------------------------------------
def fit_trend(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    model = LinearRegression().fit(x[mask].values.reshape(-1, 1), y[mask].values)
    return model


eur_model = fit_trend(field["OOIP"], field["EUR"])
ip90_model = fit_trend(field["OOIP"], field["IP90"])

# Compute field residuals to establish the baseline distribution
field["EUR_pred"] = eur_model.predict(field["OOIP"].values.reshape(-1, 1))
field["EUR_resid"] = field["EUR"] - field["EUR_pred"]

# Use FIELD residual distribution for thresholds
field_resid_std = field["EUR_resid"].std()

st.caption(
    f"EUR trend: {eur_model.coef_[0]:.4f} × OOIP + {eur_model.intercept_:.1f}  |  "
    f"Field residual σ = {field_resid_std:,.0f}"
)

# ---------------------------------------------------------------------------
# 5. SCORE PROSPECTS AGAINST FIELD TREND
# ---------------------------------------------------------------------------
pros = prospects.rename(columns={
    p_ooip: "SectionOOIP",
    p_eur: "Projected EUR",
    p_ip90: "Projected IP90"
})

if p_uwi:
    pros = pros.rename(columns={p_uwi: "UWI"})
else:
    pros["UWI"] = pros.index.astype(str)

pros.dropna(subset=["SectionOOIP", "Projected EUR", "Projected IP90"], inplace=True)

pros["Predicted EUR"] = eur_model.predict(pros["SectionOOIP"].values.reshape(-1, 1))
pros["Predicted IP90"] = ip90_model.predict(pros["SectionOOIP"].values.reshape(-1, 1))
pros["EUR Residual"] = pros["Projected EUR"] - pros["Predicted EUR"]
pros["Efficiency"] = pros["Projected EUR"] / pros["SectionOOIP"]

# ---------------------------------------------------------------------------
# 6. CLASSIFICATION
# ---------------------------------------------------------------------------
st.subheader("Classification Settings")
threshold = st.slider(
    "Residual threshold (× field σ)",
    min_value=0.1, max_value=2.0, value=0.5, step=0.1,
    help="How many field standard deviations from trend to call above/below."
)

cutoff = threshold * field_resid_std


def classify(resid):
    if resid > cutoff:
        return "Above Trend"
    elif resid < -cutoff:
        return "Below Trend"
    else:
        return "On Trend"


pros["Classification"] = pros["EUR Residual"].apply(classify)

st.subheader("Classified Prospects")
display_cols = [
    "UWI", "SectionOOIP", "Projected EUR",
    "Predicted EUR", "EUR Residual",
    "Efficiency", "Classification"
]
st.dataframe(pros[display_cols], use_container_width=True)

# Summary
st.subheader("Summary")
summary = pros["Classification"].value_counts().reset_index()
summary.columns = ["Classification", "Count"]
st.dataframe(summary, use_container_width=True)

# ---------------------------------------------------------------------------
# 7. PLOTS
# ---------------------------------------------------------------------------
x_lo = min(field["OOIP"].min(), pros["SectionOOIP"].min()) * 0.9
x_hi = max(field["OOIP"].max(), pros["SectionOOIP"].max()) * 1.1
x_trend = np.linspace(x_lo, x_hi, 200)

color_map = {
    "Above Trend": "#2ca02c",
    "On Trend": "#1f77b4",
    "Below Trend": "#d62728"
}

col1, col2 = st.columns(2)

with col1:
    fig1 = px.scatter(
        pros, x="SectionOOIP", y="Projected EUR",
        color="Classification",
        color_discrete_map=color_map,
        hover_data=["UWI"],
        title="Projected EUR vs Section OOIP",
    )
    fig1.add_trace(go.Scatter(
        x=field["OOIP"],
        y=field["EUR"],
        mode="markers",
        name="Field Wells",
        marker=dict(color="lightgrey", size=6,
                    line=dict(width=0.5, color="grey")),
    ))
    fig1.add_trace(go.Scatter(
        x=x_trend,
        y=eur_model.predict(x_trend.reshape(-1, 1)),
        mode="lines",
        name="Field Trend",
        line=dict(dash="dash", color="black"),
    ))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.scatter(
        pros, x="SectionOOIP", y="EUR Residual",
        color="Classification",
        color_discrete_map=color_map,
        hover_data=["UWI"],
        title="EUR Residual vs Section OOIP",
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="black")
    fig2.add_hline(y=cutoff, line_dash="dot",
                   line_color="green",
                   annotation_text=f"+{threshold}σ")
    fig2.add_hline(y=-cutoff, line_dash="dot",
                   line_color="red",
                   annotation_text=f"-{threshold}σ")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# 8. DOWNLOAD
# ---------------------------------------------------------------------------
st.download_button(
    label="📥 Download Classified Prospects",
    data=pros[display_cols].to_csv(index=False),
    file_name="classified_prospects.csv",
    mime="text/csv",
)
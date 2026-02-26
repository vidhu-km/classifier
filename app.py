import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("Viewfield Bakken Well Classification Tool")

# ---------------------------------------------------------------------------
# 1. DATA LOADING — Local tables.xlsx in same folder
# ---------------------------------------------------------------------------

FILE_PATH = "tables.xlsx"

try:
    sheet_names = pd.ExcelFile(FILE_PATH).sheet_names
    st.success("Loaded tables.xlsx from local folder.")
    st.caption(f"Detected sheets: {sheet_names}")
except Exception as e:
    st.error(f"Could not load tables.xlsx: {e}")
    st.stop()

try:
    field_eur = pd.read_excel(FILE_PATH, sheet_name=0)
    field_ip90 = pd.read_excel(FILE_PATH, sheet_name=1)
    field_1y = pd.read_excel(FILE_PATH, sheet_name=2)
    prospects = pd.read_excel(FILE_PATH, sheet_name=3)
except Exception as e:
    st.error(f"Error reading Excel file: {e}")
    st.stop()
# ---------------------------------------------------------------------------
# 2. READ & VALIDATE SHEETS
# ---------------------------------------------------------------------------
try:
    sheet_names = pd.ExcelFile(uploaded_file).sheet_names
    st.caption(f"Detected sheets: {sheet_names}")

    field_eur = pd.read_excel(uploaded_file, sheet_name=0)
    field_ip90 = pd.read_excel(uploaded_file, sheet_name=1)
    field_1y = pd.read_excel(uploaded_file, sheet_name=2)
    prospects = pd.read_excel(uploaded_file, sheet_name=3)
except Exception as e:
    st.error(f"Error reading Excel file: {e}")
    st.stop()


def show_columns(label, df):
    st.caption(f"**{label}** columns: {list(df.columns)}")


show_columns("Sheet 1 (EUR)", field_eur)
show_columns("Sheet 2 (IP90)", field_ip90)
show_columns("Sheet 3 (1Y)", field_1y)
show_columns("Sheet 4 (Prospects)", prospects)

# ---------------------------------------------------------------------------
# 3. FLEXIBLE COLUMN MATCHING
#    Finds the right column by looking for keywords so you don't need exact names.
# ---------------------------------------------------------------------------

def find_col(df, keywords, exclude=None):
    """Return the first column whose name contains ANY of the keywords
    (case-insensitive), optionally excluding columns matching `exclude`."""
    for col in df.columns:
        col_lower = col.lower()
        if exclude and any(ex.lower() in col_lower for ex in exclude):
            continue
        if any(kw.lower() in col_lower for kw in keywords):
            return col
    return None


# --- Sheet 1: Section, OOIP, EUR ---
sec_col_eur = find_col(field_eur, ["section"], exclude=["ooip", "eur", "ip", "cuml"])
ooip_col = find_col(field_eur, ["ooip"])
eur_col = find_col(field_eur, ["eur"])

# --- Sheet 2: Section, IP90 ---
sec_col_ip90 = find_col(field_ip90, ["section"], exclude=["ip90", "ip", "eur"])
ip90_col = find_col(field_ip90, ["ip90", "ip 90", "average section ip90"])

# --- Sheet 3: Section, 1Y Cuml ---
sec_col_1y = find_col(field_1y, ["section"], exclude=["cuml", "1y"])
y1_col = find_col(field_1y, ["1y", "cuml", "cumul"])

missing = []
for name, val in [
    ("Section (EUR sheet)", sec_col_eur),
    ("OOIP", ooip_col),
    ("EUR", eur_col),
    ("Section (IP90 sheet)", sec_col_ip90),
    ("IP90", ip90_col),
    ("Section (1Y sheet)", sec_col_1y),
    ("1Y Cuml", y1_col),
]:
    if val is None:
        missing.append(name)

if missing:
    st.error(f"Could not auto-detect these columns: {missing}. Check your sheet headers.")
    st.stop()

# --- Prospect columns ---
p_ooip = find_col(prospects, ["ooip", "section ooip", "sectionooip"])
p_eur = find_col(prospects, ["eur", "projected eur"])
p_ip90 = find_col(prospects, ["ip90", "projected ip90"])
p_uwi = find_col(prospects, ["uwi", "well", "name"])

for name, val in [
    ("Prospect OOIP", p_ooip),
    ("Prospect EUR", p_eur),
    ("Prospect IP90", p_ip90),
]:
    if val is None:
        missing.append(name)

if missing:
    st.error(f"Could not auto-detect prospect columns: {missing}.")
    st.stop()

# ---------------------------------------------------------------------------
# 4. MERGE FIELD DATA
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

# Drop rows with missing values in key columns
field.dropna(subset=["OOIP", "EUR", "IP90", "1Y"], inplace=True)

st.subheader("Field Data Preview")
st.dataframe(field, use_container_width=True)

# ---------------------------------------------------------------------------
# 5. FIT TREND MODELS
# ---------------------------------------------------------------------------

def fit_trend(x, y):
    """Fit simple linear regression, returning the model."""
    mask = np.isfinite(x) & np.isfinite(y)
    model = LinearRegression()
    model.fit(x[mask].values.reshape(-1, 1), y[mask].values)
    return model


eur_model = fit_trend(field["OOIP"], field["EUR"])
ip90_model = fit_trend(field["OOIP"], field["IP90"])
y1_model = fit_trend(field["OOIP"], field["1Y"])

st.caption(
    f"EUR trend:  EUR = {eur_model.coef_[0]:.4f} × OOIP + {eur_model.intercept_:.1f}  "
    f"| IP90 trend: IP90 = {ip90_model.coef_[0]:.4f} × OOIP + {ip90_model.intercept_:.1f}"
)

# ---------------------------------------------------------------------------
# 6. PROSPECT PREDICTIONS & RESIDUALS
# ---------------------------------------------------------------------------

# Rename prospect columns for consistency
prospects = prospects.rename(
    columns={
        p_ooip: "SectionOOIP",
        p_eur: "Projected EUR",
        p_ip90: "Projected IP90",
    }
)
if p_uwi:
    prospects = prospects.rename(columns={p_uwi: "UWI"})
else:
    prospects["UWI"] = prospects.index.astype(str)

prospects.dropna(subset=["SectionOOIP", "Projected EUR", "Projected IP90"], inplace=True)

prospects["Predicted EUR"] = eur_model.predict(
    prospects["SectionOOIP"].values.reshape(-1, 1)
)
prospects["Predicted IP90"] = ip90_model.predict(
    prospects["SectionOOIP"].values.reshape(-1, 1)
)
prospects["Predicted 1Y"] = y1_model.predict(
    prospects["SectionOOIP"].values.reshape(-1, 1)
)

prospects["EUR Residual"] = prospects["Projected EUR"] - prospects["Predicted EUR"]
prospects["IP90 Residual"] = prospects["Projected IP90"] - prospects["Predicted IP90"]
prospects["Efficiency"] = prospects["Projected EUR"] / prospects["SectionOOIP"]

# ---------------------------------------------------------------------------
# 7. CLASSIFICATION
# ---------------------------------------------------------------------------

ooip_p60 = field["OOIP"].quantile(0.6)
ooip_p40 = field["OOIP"].quantile(0.4)

eur_res_p60 = prospects["EUR Residual"].quantile(0.6)
eur_res_p40 = prospects["EUR Residual"].quantile(0.4)

st.caption(
    f"OOIP cutoffs — P40: {ooip_p40:,.0f}  |  P60: {ooip_p60:,.0f}   ·   "
    f"EUR-residual cutoffs — P40: {eur_res_p40:,.0f}  |  P60: {eur_res_p60:,.0f}"
)


def classify(row):
    ooip = row["SectionOOIP"]
    resid = row["EUR Residual"]

    high_ooip = ooip >= ooip_p60
    low_ooip = ooip <= ooip_p40
    mid_ooip = not high_ooip and not low_ooip

    high_perf = resid >= eur_res_p60
    low_perf = resid <= eur_res_p40
    mid_perf = not high_perf and not low_perf

    # --- High OOIP ---
    if high_ooip and high_perf:
        return "Type A – Core Sweet Spot"
    if high_ooip and low_perf:
        return "Type B – Resource Rich / Underperformer"
    if high_ooip and mid_perf:
        return "Type B – Resource Rich / Average Completion"

    # --- Low OOIP ---
    if low_ooip and high_perf:
        return "Type C – Completion Driven Outperformer"
    if low_ooip and mid_perf:
        return "Type D – Small but Efficient"
    if low_ooip and low_perf:
        return "Type E – Marginal / Edge"

    # --- Mid OOIP ---
    if mid_ooip and high_perf:
        return "Type A– – Near-Core Outperformer"
    if mid_ooip and mid_perf:
        return "Type D – Average"
    if mid_ooip and low_perf:
        return "Type F – Interference / Anomaly"

    return "Unclassified"


prospects["Type"] = prospects.apply(classify, axis=1)

st.subheader("Prospect Classification")
st.dataframe(
    prospects[
        [
            "UWI",
            "SectionOOIP",
            "Projected EUR",
            "Predicted EUR",
            "Efficiency",
            "EUR Residual",
            "Type",
        ]
    ],
    use_container_width=True,
)

# Summary counts
st.subheader("Classification Summary")
st.dataframe(
    prospects["Type"].value_counts().reset_index().rename(
        columns={"index": "Type", "Type": "Count"}
    ),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# 8. PLOTS
# ---------------------------------------------------------------------------

# Helper: generate trend-line trace for overlay
def trend_line_trace(model, x_range, name="Field Trend"):
    x_arr = np.linspace(x_range[0], x_range[1], 200)
    y_arr = model.predict(x_arr.reshape(-1, 1))
    return go.Scatter(
        x=x_arr, y=y_arr, mode="lines",
        name=name, line=dict(dash="dash", color="black", width=2),
    )


x_lo = min(field["OOIP"].min(), prospects["SectionOOIP"].min()) * 0.9
x_hi = max(field["OOIP"].max(), prospects["SectionOOIP"].max()) * 1.1

col1, col2 = st.columns(2)

with col1:
    fig1 = px.scatter(
        prospects,
        x="SectionOOIP",
        y="Projected EUR",
        color="Type",
        hover_data=["UWI"],
        title="Projected EUR vs Section OOIP",
    )
    # Overlay field data as grey dots
    fig1.add_trace(
        go.Scatter(
            x=field["OOIP"], y=field["EUR"],
            mode="markers", name="Field Wells",
            marker=dict(color="lightgrey", size=6, line=dict(width=0.5, color="grey")),
        )
    )
    fig1.add_trace(trend_line_trace(eur_model, [x_lo, x_hi], "EUR Trend"))
    fig1.update_layout(xaxis_title="Section OOIP", yaxis_title="EUR")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.scatter(
        prospects,
        x="SectionOOIP",
        y="Projected IP90",
        color="Type",
        hover_data=["UWI"],
        title="Projected IP90 vs Section OOIP",
    )
    fig2.add_trace(
        go.Scatter(
            x=field["OOIP"], y=field["IP90"],
            mode="markers", name="Field Wells",
            marker=dict(color="lightgrey", size=6, line=dict(width=0.5, color="grey")),
        )
    )
    fig2.add_trace(trend_line_trace(ip90_model, [x_lo, x_hi], "IP90 Trend"))
    fig2.update_layout(xaxis_title="Section OOIP", yaxis_title="IP90")
    st.plotly_chart(fig2, use_container_width=True)

# Residual plot
fig3 = px.scatter(
    prospects,
    x="SectionOOIP",
    y="EUR Residual",
    color="Type",
    hover_data=["UWI"],
    title="EUR Residual vs Section OOIP",
)
fig3.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Trend = 0")
fig3.add_hline(y=eur_res_p60, line_dash="dot", line_color="green", annotation_text="P60")
fig3.add_hline(y=eur_res_p40, line_dash="dot", line_color="red", annotation_text="P40")
fig3.update_layout(xaxis_title="Section OOIP", yaxis_title="EUR Residual")
st.plotly_chart(fig3, use_container_width=True)

# Efficiency plot
fig4 = px.scatter(
    prospects,
    x="SectionOOIP",
    y="Efficiency",
    color="Type",
    hover_data=["UWI"],
    title="Recovery Efficiency (EUR / OOIP) vs Section OOIP",
)
fig4.update_layout(xaxis_title="Section OOIP", yaxis_title="Efficiency (EUR / OOIP)")
st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------------------------------
# 9. DOWNLOAD
# ---------------------------------------------------------------------------
st.download_button(
    label="📥 Download Classified Prospects CSV",
    data=prospects.to_csv(index=False),
    file_name="classified_prospects.csv",
    mime="text/csv",
)
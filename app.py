import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
import numpy as np

st.title("Bakken Prospect Classification (Simplified)")

# --- Upload Excel file ---
file = st.file_uploader("Upload tables.xlsx", type=["xlsx"])
if not file:
    st.stop()

# --- Load field and prospect data (expected 2 sheets only) ---
try:
    field = pd.read_excel(file, sheet_name="Field")
    prospects = pd.read_excel(file, sheet_name="Prospects")
except Exception as e:
    st.error(f"Error reading file — ensure 'Field' and 'Prospects' sheets exist.\n\n{e}")
    st.stop()

# --- Ensure correct columns exist ---
required_field_cols = ["OOIP", "EUR"]
required_prospect_cols = ["SectionOOIP", "Projected EUR"]

for c in required_field_cols:
    if c not in field.columns:
        st.error(f"Missing '{c}' in Field sheet.")
        st.stop()

for c in required_prospect_cols:
    if c not in prospects.columns:
        st.error(f"Missing '{c}' in Prospects sheet.")
        st.stop()

# --- Fit simple trend (EUR vs OOIP) ---
model = LinearRegression()
x = field["OOIP"].values.reshape(-1, 1)
y = field["EUR"].values
model.fit(x, y)

prospects["Predicted EUR"] = model.predict(prospects["SectionOOIP"].values.reshape(-1, 1))
prospects["Residual"] = prospects["Projected EUR"] - prospects["Predicted EUR"]

# --- Classify by residuals ---
p40, p60 = prospects["Residual"].quantile([0.4, 0.6])

def classify(res):
    if res >= p60:
        return "Above Trend"
    elif res <= p40:
        return "Below Trend"
    return "Average"

prospects["Class"] = prospects["Residual"].apply(classify)

# --- Display summary ---
st.subheader("Prospect Classification Summary")
st.dataframe(prospects)

st.bar_chart(prospects["Class"].value_counts())

# --- Plot ---
fig = px.scatter(
    prospects,
    x="SectionOOIP",
    y="Projected EUR",
    color="Class",
    trendline="ols",
    title="Projected EUR vs OOIP (with Field Trend)",
)
fig.add_scatter(x=field["OOIP"], y=field["EUR"], mode="markers", name="Field Wells", marker=dict(color="gray"))
st.plotly_chart(fig, use_container_width=True)

# --- Download ---
st.download_button(
    "Download Classified Prospects CSV",
    prospects.to_csv(index=False),
    "classified_prospects.csv",
    "text/csv",
)
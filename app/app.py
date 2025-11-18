import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Data
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/synthetic_transaction_data.csv", parse_dates=["date"])

txn_data = load_data()

# -------------------------------
# 2. Sidebar Controls
# -------------------------------
st.sidebar.title("Forecast Settings")

corridor = st.sidebar.selectbox(
    "Select Corridor",
    txn_data["corridor"].unique()
)

fees = st.sidebar.multiselect(
    "Select Fee Rates",
    [0.015, 0.02, 0.025],
    default=[0.02]
)

forecast_years = st.sidebar.slider("Forecast Horizon (Years)", 1, 5, 3)

# -------------------------------
# 3. Filter Data
# -------------------------------
df_corridor = txn_data[txn_data["corridor"] == corridor]

# -------------------------------
# 4. Prophet Forecasting (baseline: 2% fee)
# -------------------------------
df = df_corridor.rename(columns={"date":"ds","net_revenue_0.02":"y"})
model = Prophet(yearly_seasonality=True, daily_seasonality=False)
model.fit(df)

future = model.make_future_dataframe(periods=365*forecast_years)
forecast = model.predict(future)

# -------------------------------
# 5. Scenario Analysis
# -------------------------------
results = {}
for fee in fees:
    col_name = f"net_revenue_{fee}"
    if col_name in df_corridor.columns:
        forecast[col_name] = forecast["yhat"] * (fee/0.02)
        results[fee] = forecast[col_name].sum()

# -------------------------------
# 6. Display Results
# -------------------------------
st.title("FX & Transaction Forecasting Tool")
st.subheader(f"Corridor: {corridor}")

# Safe column selection for chart
available_cols = [f"net_revenue_{fee}" for fee in fees if f"net_revenue_{fee}" in df_corridor.columns]

if available_cols:
    st.write("### Historical Net Revenue")
    st.line_chart(df_corridor.set_index("date")[available_cols])
else:
    st.warning("No matching fee scenario columns found in dataset.")

st.write("### Forecasted Net Revenue")
fig, ax = plt.subplots(figsize=(10,5))
for fee in fees:
    col_name = f"net_revenue_{fee}"
    if col_name in forecast.columns:
        ax.plot(forecast["ds"], forecast[col_name], label=f"{fee*100:.1f}% Fee")
ax.set_title(f"{forecast_years}-Year Forecast - {corridor}")
ax.set_xlabel("Date")
ax.set_ylabel("Net Revenue (USD)")
ax.legend()
st.pyplot(fig)

st.write("### Scenario Results (Total Net Revenue)")
for fee, total in results.items():
    st.write(f"{fee*100:.1f}% fee: ${total:,.0f}")

import streamlit as st
import pandas as pd
from forecasting import forecast_fx
from scenario_analysis import fee_scenarios

st.title("FX & Revenue Forecasting Tool")

fx_data = pd.read_csv("data/fx_rates.csv", parse_dates=["date"])
forecast = forecast_fx(fx_data)

st.line_chart(forecast[["ds","yhat"]].set_index("ds"))

fees = st.multiselect("Select fee rates", [0.015,0.02,0.025], default=[0.02])
results = fee_scenarios(forecast, fees=fees)

st.write("Scenario Results:", results)

from prophet import Prophet

def forecast_fx(df, periods=90):
    df = df.rename(columns={"date":"ds","rate":"y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

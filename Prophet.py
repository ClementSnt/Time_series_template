import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


df = pd.read_excel("link to database with columns date and volumes")
holidays = pd.read_excel("link to database with just one column date with dates of holidays only")


holidays['ds'] = pd.to_datetime(holidays['Date'])
holidays['holiday'] = 'holidays'
holidays = holidays[['ds', 'holiday']]


df['ds'] = pd.to_datetime(df['Date'])
df['y'] = df['volumes']
df = df[['ds', 'y']]


m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    holidays=holidays
)

m.fit(df)


future = m.make_future_dataframe(periods=365, freq='D') # for daily forecast for the next 365 days after the last day in the training database
forecast = m.predict(future)


fig1 = m.plot(forecast)
plt.show()

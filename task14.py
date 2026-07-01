import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("AirPassengers.csv")

df['Month'] = pd.to_datetime(df['Month'])

df.set_index('Month', inplace=True)

print("First 5 Rows:")
print(df.head())
plt.figure(figsize=(10,5))
plt.plot(df['#Passengers'])
plt.title("AirPassengers Dataset")
plt.xlabel("Year")
plt.ylabel("#Passengers")
plt.show()
decomposition = seasonal_decompose(df['#Passengers'], model='multiplicative', period=12)
decomposition.plot()
plt.show()
result = adfuller(df['#Passengers'])

print("\nADF Test Result")
print("ADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] < 0.05:
    print("Data is Stationary")
else:
    print("Data is Not Stationary")
model = ARIMA(df['#Passengers'], order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=12)
print("\nForecast for Next 12 Months:")
print(forecast)
plt.figure(figsize=(10,5))
plt.plot(df['#Passengers'], label="Original Data")
plt.plot(forecast.index, forecast, color="red", label="Forecast")
plt.title("ARIMA Forecast")
plt.legend()
plt.show()
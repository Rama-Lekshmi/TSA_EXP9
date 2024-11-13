# Developed By : Rama E.K. Lekshmi
# Register Number : 212222240082

# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```py
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima

df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
df.set_index('Month', inplace=True)
df.rename(columns={'#Passengers': 'passengers'}, inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(df['passengers'])
plt.title('Time Series Plot of Air Passengers')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.show()

def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] < 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary.")

adf_test(df['passengers'])

plot_acf(df['passengers'], lags=30)
plot_pacf(df['passengers'], lags=30)
plt.show()

df['passengers_diff'] = df['passengers'].diff().dropna()
plt.figure(figsize=(12, 6))
plt.plot(df['passengers_diff'])
plt.title('Differenced Time Series of Air Passengers')
plt.xlabel('Date')
plt.ylabel('Differenced Passengers')
plt.show()

adf_test(df['passengers_diff'].dropna())

auto_model = auto_arima(df['passengers'], seasonal=False, stepwise=True, trace=True)
print(auto_model.summary())
p, d, q = auto_model.order

model = ARIMA(df['passengers'], order=(p, d, q))
model_fit = model.fit()

print(model_fit.summary())

forecast = model_fit.forecast(steps=10)
print("Forecasted values:\n", forecast)

plt.figure(figsize=(12, 6))
plt.plot(df['passengers'], label='Original Data')
plt.plot(forecast.index, forecast, color='red', label='Forecast', marker='o')
plt.title('ARIMA Model Forecast for Air Passengers')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()

train_size = int(len(df) * 0.8)
train, test = df['passengers'][:train_size], df['passengers'][train_size:]

model = ARIMA(train, order=(p, d, q))
model_fit = model.fit()
predictions = model_fit.forecast(steps=len(test))

mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')


plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual Data')
plt.plot(test.index, predictions, color='red', label='Predicted Data')
plt.title('Actual vs Predicted for Air Passengers')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()

```

### OUTPUT:

![image](https://github.com/user-attachments/assets/e5f241d9-7c59-414a-b1a0-ac9223b7b6e2)

![image](https://github.com/user-attachments/assets/1bb110e7-2f6e-48c7-9eaa-65167ba8ae81)

![image](https://github.com/user-attachments/assets/b44e2810-9d0e-496c-807b-c8077ddfe4b8)

![image](https://github.com/user-attachments/assets/56217eb9-887b-4433-b83f-6fbf8c1a4f73)

![image](https://github.com/user-attachments/assets/f4658261-a42e-4f39-9afd-d93b30151973)

![image](https://github.com/user-attachments/assets/81de1239-8423-4ee1-8e11-db927c3a957c)


### RESULT:
Thus the program run successfully based on the ARIMA model using python.

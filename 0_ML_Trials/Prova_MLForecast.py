import pandas as pd
from xgboost import XGBRegressor
from mlforecast import MLForecast
from utilsforecast.plotting import plot_series
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

# Load the data
data = pd.read_csv('../datasets/ETT-small/ETTh1.csv', parse_dates=['date'])
data['unique_id'] = 1
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')
data.drop(['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'], axis=1, inplace=True)
data.rename(columns={'date': 'ds', 'OT': 'y'}, inplace=True)

# Split the data
train = data[:8640]
test = data[8640:]

# Create the forecasting model
models = [LGBMRegressor(verbosity=-1),
    XGBRegressor(),
    RandomForestRegressor(random_state=0)]

fcst = MLForecast(
    models=models,
    freq='H',
    lags=[7, 14, 21, 28]
)

# Fit the model
fcst.fit(train)

# Forecast for multiple sequence lengths and compute metrics
seq_lengths = [96, 192, 336, 720]
results = []

for length in seq_lengths:
    predictions = fcst.predict(length)

    y_values = test['y'].tolist()[:length]
    pred_values = predictions['XGBRegressor'].tolist()[:length]

    # Convert lists to numpy arrays
    pred_values_array = np.array(pred_values)
    y_values_array = np.array(y_values)

    # Compute mse and mae
    mse = ((pred_values_array - y_values_array) ** 2).mean()
    mae = abs(pred_values_array - y_values_array).mean()

    results.append({
        'seq_length': length,
        'mse': mse,
        'mae': mae
    })

# Print the results
for result in results:
    print(f"Sequence length: {result['seq_length']}, MSE: {result['mse']}, MAE: {result['mae']}")

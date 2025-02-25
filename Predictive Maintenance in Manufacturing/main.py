# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
train_data = pd.read_csv('train_FD001.txt', sep=" ", header=None)
test_data = pd.read_csv('test_FD001.txt', sep=" ", header=None)
rul_data = pd.read_csv('RUL_FD001.txt', sep=" ", header=None)

# Drop unnecessary columns (last two columns are NaN)
train_data.drop(columns=[26, 27], inplace=True)
test_data.drop(columns=[26, 27], inplace=True)

# Check the structure of rul_data
print(rul_data.head())

# If rul_data has two columns, drop the first column (index 0)
if rul_data.shape[1] == 2:
    rul_data.drop(columns=[0], inplace=True)

# Add column names for clarity
columns = ['engine_id', 'cycle'] + [f'sensor_{i}' for i in range(1, 22)] + ['setting_1', 'setting_2', 'setting_3']
train_data.columns = columns
test_data.columns = columns


# Add RUL (Remaining Useful Life) to the training data
def add_rul(df):
    max_cycles = df.groupby('engine_id')['cycle'].transform('max')
    df['RUL'] = max_cycles - df['cycle']
    return df


train_data = add_rul(train_data)


# Feature Engineering
def feature_engineering(df):
    # Rolling mean for sensor data
    for sensor in [f'sensor_{i}' for i in range(1, 22)]:
        df[f'{sensor}_rolling_mean'] = df.groupby('engine_id')[sensor].transform(lambda x: x.rolling(window=5).mean())

    # Rate of change for sensor data
    for sensor in [f'sensor_{i}' for i in range(1, 22)]:
        df[f'{sensor}_rate_of_change'] = df.groupby('engine_id')[sensor].diff().fillna(0)

    return df


train_data = feature_engineering(train_data)
test_data = feature_engineering(test_data)

# Drop rows with NaN values (due to rolling window)
train_data.dropna(inplace=True)

# Normalize the data
scaler = MinMaxScaler()
sensor_columns = [f'sensor_{i}' for i in range(1, 22)] + [f'sensor_{i}_rolling_mean' for i in range(1, 22)] + [
    f'sensor_{i}_rate_of_change' for i in range(1, 22)]
train_data[sensor_columns] = scaler.fit_transform(train_data[sensor_columns])
test_data[sensor_columns] = scaler.transform(test_data[sensor_columns])

# Prepare features and target
X = train_data.drop(columns=['engine_id', 'cycle', 'RUL'])
y = train_data['RUL']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Predict RUL for test data
test_data['predicted_RUL'] = model.predict(test_data.drop(columns=['engine_id', 'cycle']))

# Save predictions
test_data[['engine_id', 'cycle', 'predicted_RUL']].to_csv('test_predictions.csv', index=False)
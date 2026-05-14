# FLASHFLOOD_ML
Hydrological flash flood analysis using machine learrning models
1. RANDOM FOREST
```python
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

# =========================================
# LOAD DATA
# =========================================

files = [
    "2021_Original.csv",
    "2022_Original.csv",
    "2023_Original.csv"
]
for file in files:
    if not os.path.exists(file):
        print(f"Warning: {file} not found.")
        continue

    print(f"\n" + "="*40)
    print(f"Processing File: {file}")
    print("="*40)

    # Load Data
    df = pd.read_csv(file)
# =========================================
# COLUMN NAMES
# =========================================

rain_col = "Rainfall_mm"
target_col = "Observed"

# =========================================
# EXTRACT DATA
# =========================================

X = df[[rain_col]].values
y = df[[target_col]].values

# =========================================
# NORMALIZATION
# =========================================

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_norm = x_scaler.fit_transform(X)
y_norm = y_scaler.fit_transform(y)

# =========================================
# TRAIN-TEST SPLIT (80:20)
# =========================================

split_index = int(len(df) * 0.8)

X_train = X_norm[:split_index]
X_test = X_norm[split_index:]

y_train = y_norm[:split_index].ravel()
y_test = y_norm[split_index:].ravel()

# =========================================
# RANDOM FOREST MODEL
# =========================================

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

rf.fit(X_train, y_train)

# =========================================
# PREDICTIONS (NORMALIZED)
# =========================================

y_pred_norm = rf.predict(X_norm)

# =========================================
# DENORMALIZATION
# =========================================

y_pred_denorm = y_scaler.inverse_transform(
    y_pred_norm.reshape(-1, 1)
).ravel()

y_actual_denorm = y.ravel()

rain_denorm = X.ravel()
rain_norm = X_norm.ravel()

# =========================================
# PERFORMANCE METRICS
# =========================================

# PCC
pcc, _ = pearsonr(y_actual_denorm, y_pred_denorm)

# NSE
nse = 1 - (
    np.sum((y_actual_denorm - y_pred_denorm) ** 2)
    /
    np.sum((y_actual_denorm - np.mean(y_actual_denorm)) ** 2)
)

# PBIAS
pbias = (
    np.sum(y_actual_denorm - y_pred_denorm)
    /
    np.sum(y_actual_denorm)
) * 100

# =========================================
# PRINT METRICS
# =========================================

print("\n===== RANDOM FOREST PERFORMANCE =====\n")

print(f"PCC   : {pcc:.4f}")
print(f"NSE   : {nse:.4f}")
print(f"PBIAS : {pbias:.4f}")

# =========================================
# SAVE COMPLETE CSV
# =========================================

results = pd.DataFrame({

    # Rainfall
    "Rainfall_Original": rain_denorm,
    "Rainfall_Normalized": rain_norm,

    # Discharge
    "Observed_Discharge": y_actual_denorm,
    "Observed_Discharge_Normalized": y_norm.ravel(),

    # Prediction
    "Predicted_Discharge": y_pred_denorm,
    "Predicted_Discharge_Normalized": y_pred_norm
})

year_name = file.split("_")[0]
    output_file = f"RF_Results_{year_name}.csv"

    results.to_csv(output_file, index=False)
    print(f"Exported: {output_file}")
```
2. SUPPORT VECTOR MACHINE(SVM)
```python
import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

# =========================================
# LOAD DATA
# =========================================

files = [
    "2021_Original.csv",
    "2022_Original.csv",
    "2023_Original.csv"
]
for file in files:
    if not os.path.exists(file):
        print(f"Warning: {file} not found.")
        continue

    print(f"\n" + "="*45)
    print(f"Processing File: {file}")
    print("="*45)

    # Load Data
    df = pd.read_csv(file)

# =========================================
# COLUMN NAMES
# =========================================

rain_col = "Rainfall_mm"
target_col = "Observed"

# =========================================
# EXTRACT DATA
# =========================================

X = df[[rain_col]].values
y = df[[target_col]].values

# =========================================
# NORMALIZATION
# =========================================

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_norm = x_scaler.fit_transform(X)
y_norm = y_scaler.fit_transform(y)

# =========================================
# TRAIN-TEST SPLIT (80:20)
# =========================================

split_index = int(len(df) * 0.8)

X_train = X_norm[:split_index]
X_test = X_norm[split_index:]

y_train = y_norm[:split_index].ravel()
y_test = y_norm[split_index:].ravel()

# =========================================
# SVM MODEL
# =========================================

svm = SVR(
    kernel='rbf',
    C=10,
    gamma=0.5,
    epsilon=0.01
)

svm.fit(X_train, y_train)

# =========================================
# PREDICTIONS (NORMALIZED)
# =========================================

y_pred_norm = svm.predict(X_norm)

# =========================================
# DENORMALIZATION
# =========================================

y_pred_denorm = y_scaler.inverse_transform(
    y_pred_norm.reshape(-1, 1)
).ravel()

y_actual_denorm = y.ravel()

rain_denorm = X.ravel()
rain_norm = X_norm.ravel()

# =========================================
# PERFORMANCE METRICS
# =========================================

# PCC
pcc, _ = pearsonr(y_actual_denorm, y_pred_denorm)

# NSE
nse = 1 - (
    np.sum((y_actual_denorm - y_pred_denorm) ** 2)
    /
    np.sum((y_actual_denorm - np.mean(y_actual_denorm)) ** 2)
)

# PBIAS
pbias = (
    np.sum(y_actual_denorm - y_pred_denorm)
    /
    np.sum(y_actual_denorm)
) * 100

# =========================================
# PRINT METRICS
# =========================================

print("\n===== SVM PERFORMANCE =====\n")

print(f"PCC   : {pcc:.4f}")
print(f"NSE   : {nse:.4f}")
print(f"PBIAS : {pbias:.4f}")

# =========================================
# SAVE COMPLETE CSV
# =========================================

results = pd.DataFrame({

    # Rainfall
    "Rainfall_Original": rain_denorm,
    "Rainfall_Normalized": rain_norm,

    # Observed discharge
    "Observed_Discharge": y_actual_denorm,
    "Observed_Discharge_Normalized": y_norm.ravel(),

    # Predicted discharge
    "Predicted_Discharge": y_pred_denorm,
    "Predicted_Discharge_Normalized": y_pred_norm
})

year_name = file.split("_")[0]
    output_file = f"SVM_Results_{year_name}.csv"

    results.to_csv(output_file, index=False)
    print(f"Successfully saved to: {output_file}")
```
3. GRADIENT BOOSTING MACHINE(GBM)
```python
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

# =========================================
# LOAD DATA
# =========================================

files = [
    "2021_Original.csv",
    "2022_Original.csv",
    "2023_Original.csv"
]
for file in files:
    if not os.path.exists(file):
        print(f"Warning: {file} not found.")
        continue

    print(f"\n" + "="*45)
    print(f"Processing File: {file}")
    print("="*45)

    # Load Data
    df = pd.read_csv(file)
# =========================================
# COLUMN NAMES
# =========================================

rain_col = "Rainfall_mm"
target_col = "Observed"

# =========================================
# EXTRACT DATA
# =========================================

X = df[[rain_col]].values
y = df[[target_col]].values

# =========================================
# NORMALIZATION
# =========================================

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_norm = x_scaler.fit_transform(X)
y_norm = y_scaler.fit_transform(y)

# =========================================
# TRAIN-TEST SPLIT (80:20)
# =========================================

split_index = int(len(df) * 0.8)

X_train = X_norm[:split_index]
X_test = X_norm[split_index:]

y_train = y_norm[:split_index].ravel()
y_test = y_norm[split_index:].ravel()

# =========================================
# GBM MODEL
# =========================================

gbm = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gbm.fit(X_train, y_train)

# =========================================
# PREDICTIONS (NORMALIZED)
# =========================================

y_pred_norm = gbm.predict(X_norm)

# =========================================
# DENORMALIZATION
# =========================================

y_pred_denorm = y_scaler.inverse_transform(
    y_pred_norm.reshape(-1, 1)
).ravel()

y_actual_denorm = y.ravel()

rain_denorm = X.ravel()
rain_norm = X_norm.ravel()

# =========================================
# PERFORMANCE METRICS
# =========================================

# PCC
pcc, _ = pearsonr(y_actual_denorm, y_pred_denorm)

# NSE
nse = 1 - (
    np.sum((y_actual_denorm - y_pred_denorm) ** 2)
    /
    np.sum((y_actual_denorm - np.mean(y_actual_denorm)) ** 2)
)

# PBIAS
pbias = (
    np.sum(y_actual_denorm - y_pred_denorm)
    /
    np.sum(y_actual_denorm)
) * 100

# =========================================
# PRINT METRICS
# =========================================

print("\n===== GBM PERFORMANCE =====\n")

print(f"PCC   : {pcc:.4f}")
print(f"NSE   : {nse:.4f}")
print(f"PBIAS : {pbias:.4f}")

# =========================================
# SAVE COMPLETE CSV
# =========================================

results = pd.DataFrame({

    # Rainfall
    "Rainfall_Original": rain_denorm,
    "Rainfall_Normalized": rain_norm,

    # Observed discharge
    "Observed_Discharge": y_actual_denorm,
    "Observed_Discharge_Normalized": y_norm.ravel(),

    # Predicted discharge
    "Predicted_Discharge": y_pred_denorm,
    "Predicted_Discharge_Normalized": y_pred_norm
})
year_name = file.split("_")[0]
    output_file = f"GBM_Results_{year_name}.csv"

    results.to_csv(output_file, index=False)
    print(f"Successfully saved results to: {output_file}")
```
4. LONG TERM SHORT TERM MEMORY(LSTM)
```python
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =========================================
# LOAD DATA
# =========================================

files = [
    "2021_Original.csv",
    "2022_Original.csv",
    "2023_Original.csv"
]
for file in files:
    if not os.path.exists(file):
        print(f"Warning: {file} not found.")
        continue

    print(f"\n" + "="*45)
    print(f"Processing Deep Learning Model: {file}")
    print("="*45)

    # Load Data
    df = pd.read_csv(file)

# =========================================
# COLUMN NAMES
# =========================================

rain_col = "Rainfall_Scattered"
target_col = "Observed"

# =========================================
# EXTRACT DATA
# =========================================

X = df[[rain_col]].values
y = df[[target_col]].values

# =========================================
# NORMALIZATION
# =========================================

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_norm = x_scaler.fit_transform(X)
y_norm = y_scaler.fit_transform(y)

# =========================================
# CREATE SEQUENCES
# =========================================

time_steps = 1

X_seq = []
y_seq = []

for i in range(len(X_norm) - time_steps + 1):

    X_seq.append(X_norm[i:i + time_steps])
    y_seq.append(y_norm[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# =========================================
# TRAIN-TEST SPLIT (80:20)
# =========================================

split_index = int(len(X_seq) * 0.8)

X_train = X_seq[:split_index]
X_test = X_seq[split_index:]

y_train = y_seq[:split_index]
y_test = y_seq[split_index:]

# =========================================
# LSTM MODEL
# =========================================

model = Sequential()

model.add(
    LSTM(
        64,
        activation='tanh',
        input_shape=(X_train.shape[1], X_train.shape[2])
    )
)

model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mse'
)

# =========================================
# TRAIN MODEL
# =========================================

history = model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=16,
    verbose=0
    callbacks=[early_stop]
)

# =========================================
# PREDICTIONS
# =========================================

y_pred_norm_seq = model.predict(X_seq)

# =========================================
# DENORMALIZATION
# =========================================

y_pred_denorm_seq = y_scaler.inverse_transform(
    y_pred_norm_seq
).ravel()

y_actual_denorm = y.ravel()

# =========================================
# CREATE FULL LENGTH ARRAYS
# =========================================

full_pred_denorm = np.empty(len(df))
full_pred_denorm[:] = np.nan

full_pred_norm = np.empty(len(df))
full_pred_norm[:] = np.nan

# Since timestep = 1
full_pred_denorm[:] = y_pred_denorm_seq
full_pred_norm[:] = y_pred_norm_seq.ravel()

# =========================================
# PERFORMANCE METRICS
# =========================================

valid_indices = ~np.isnan(full_pred_denorm)

obs_valid = y_actual_denorm[valid_indices]
pred_valid = full_pred_denorm[valid_indices]

# PCC
pcc, _ = pearsonr(obs_valid, pred_valid)

# NSE
nse = 1 - (
    np.sum((obs_valid - pred_valid) ** 2)
    /
    np.sum((obs_valid - np.mean(obs_valid)) ** 2)
)

# PBIAS
pbias = (
    np.sum(obs_valid - pred_valid)
    /
    np.sum(obs_valid)
) * 100

# =========================================
# PRINT METRICS
# =========================================

print("\n===== LSTM PERFORMANCE =====\n")

print(f"PCC   : {pcc:.4f}")
print(f"NSE   : {nse:.4f}")
print(f"PBIAS : {pbias:.4f}")

# =========================================
# SAVE COMPLETE CSV
# =========================================

results = pd.DataFrame({

    # Rainfall
    "Rainfall_Original": X.ravel(),
    "Rainfall_Normalized": X_norm.ravel(),

    # Observed discharge
    "Observed_Discharge": y.ravel(),
    "Observed_Discharge_Normalized": y_norm.ravel(),

    # Predicted discharge
    "Predicted_Discharge": full_pred_denorm,
    "Predicted_Discharge_Normalized": full_pred_norm
})
year_name = file.split("_")[0]
    output_file = f"LSTM_Results_{year_name}.csv"
    results.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

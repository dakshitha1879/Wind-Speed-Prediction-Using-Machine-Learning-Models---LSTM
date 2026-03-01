import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. DATA DIRECTORY (ACTUAL PATH)
# ==========================================

DATA_DIR = r"C:\Users\Dell\Desktop\LSTM\data" # Change the DIR

varanasi = pd.read_csv(os.path.join(DATA_DIR, "wind_data_varanasi.csv"))
lucknow = pd.read_csv(os.path.join(DATA_DIR, "wind_data_lucknow.csv"))
trivendrum = pd.read_csv(os.path.join(DATA_DIR, "wind_data_trivendrum.csv"))
ahmedabad = pd.read_csv(os.path.join(DATA_DIR, "wind_data_ahemdabad.csv"))

# ==========================================
# 2. CLEANING FUNCTION
# ==========================================

def clean_and_prepare(df, city_id):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    df = df.set_index(date_col)

    df = df.select_dtypes(include=["number"])
    df = df.dropna()

    df["city"] = city_id
    return df

varanasi = clean_and_prepare(varanasi, 0)
lucknow = clean_and_prepare(lucknow, 1)
trivendrum = clean_and_prepare(trivendrum, 2)
ahmedabad = clean_and_prepare(ahmedabad, 3)

# ==========================================
# 3. COMBINE DATA
# ==========================================

data = pd.concat([varanasi, lucknow, trivendrum, ahmedabad])
data = data.sort_values(["city"])
data = data.reset_index(drop=True)

print("Combined Dataset shape:", data.shape)

# ==========================================
# 4. TRAIN TEST SPLIT
# ==========================================

train_size = int(len(data) * 0.8)
train = data.iloc[:train_size]
test = data.iloc[train_size:]

# ==========================================
# 5. SCALING
# ==========================================

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

target_col = "windspeed10m"

if target_col not in train.columns:
    print("Available columns:", train.columns)
    raise ValueError("Target column 'windspeed10m' not found!")

target_index = train.columns.get_loc(target_col)

# ==========================================
# 6. CREATE SEQUENCES
# ==========================================

def create_sequences(dataset, steps=24):
    X, y = [], []
    for i in range(len(dataset) - steps):
        X.append(dataset[i:i+steps])
        y.append(dataset[i+steps][target_index])
    return np.array(X), np.array(y)

timesteps = 24

X_train, y_train = create_sequences(train_scaled, timesteps)
X_test, y_test = create_sequences(test_scaled, timesteps)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# ==========================================
# 7. BUILD LSTM MODEL
# ==========================================

model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(timesteps, X_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(32))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

model.summary()

# ==========================================
# 8. TRAIN MODEL
# ==========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ==========================================
# 9. EVALUATE
# ==========================================

y_pred = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("="*40)
print("MAE  :", round(mae, 4))
print("RMSE :", round(rmse, 4))
print("R²   :", round(r2, 4))

# ==========================================
# 10. SAVE MODEL
# ==========================================

model.save(r"C:\Users\Dell\Desktop\LSTM\lstm_wind_model_4cities.h5")
print("\nModel saved as lstm_wind_model_4cities.h5")

# ==========================================
# 11. PLOT RESULTS
# ==========================================

plt.figure()
plt.plot(y_test[:200])
plt.plot(y_pred[:200])
plt.legend(["Actual", "Predicted"])
plt.title("Actual vs Predicted Wind Speed")
plt.show()
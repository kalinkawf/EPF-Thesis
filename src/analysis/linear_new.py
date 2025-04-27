import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Funkcja do obliczania metryk
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask_smape = denominator != 0
    smape = np.mean(2 * np.abs(y_pred[mask_smape] - y_true[mask_smape]) / denominator[mask_smape]) * 100 if mask_smape.sum() > 0 else np.nan
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, mape, smape, r2

# --- Wczytanie danych ---

df_full = pd.read_csv("../../data/database.csv")
df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])

df_short = pd.read_csv("../../data/short_database.csv")
df_short["timestamp"] = pd.to_datetime(df_short["timestamp"])

# --- Preprocessing dla pełnego zbioru danych ---

# 1. Uzupełnienie brakujących wartości
# df_full = df_full.interpolate(method="linear").fillna(method="ffill").fillna(method="bfill")

# 2. Kodowanie zmiennych cyklicznych (tylko sinus)
df_full["day_of_week_sin"] = np.sin(2 * np.pi * df_full["day_of_week"] / 7)
df_full["month_sin"] = np.sin(2 * np.pi * df_full["month"] / 12)
df_full["hour_sin"] = np.sin(2 * np.pi * df_full["hour"] / 24)
df_full = df_full.drop(columns=["day_of_week", "month", "hour"])

# 3. Kopia dla logarytmizacji
df_full_log = df_full.copy()
epsilon = 1e-6
df_full_log["fixing_i_price_log"] = np.log(df_full_log["fixing_i_price"] + 1 + epsilon)

# 4. Standaryzacja danych (bez logarytmizacji)
train_spokojny_full = df_full[(df_full["timestamp"].dt.year >= 2016) & (df_full["timestamp"].dt.year <= 2018)]
test_spokojny_full = df_full[df_full["timestamp"].dt.year == 2019]

columns_to_scale_full = [col for col in df_full.columns if col not in ["timestamp", "is_holiday"]]

scaler_full = StandardScaler()
train_spokojny_full_scaled = train_spokojny_full.copy()
test_spokojny_full_scaled = test_spokojny_full.copy()

train_spokojny_full_scaled[columns_to_scale_full] = scaler_full.fit_transform(train_spokojny_full[columns_to_scale_full])
test_spokojny_full_scaled[columns_to_scale_full] = scaler_full.transform(test_spokojny_full[columns_to_scale_full])

df_full = pd.concat([train_spokojny_full_scaled, test_spokojny_full_scaled])

# 5. Standaryzacja danych (z logarytmizacją)
train_spokojny_full_log = df_full_log[(df_full_log["timestamp"].dt.year >= 2016) & (df_full_log["timestamp"].dt.year <= 2018)]
test_spokojny_full_log = df_full_log[df_full_log["timestamp"].dt.year == 2019]

columns_to_scale_full_log = [col for col in df_full_log.columns if col not in ["timestamp", "is_holiday"]]

scaler_full_log = StandardScaler()
train_spokojny_full_log_scaled = train_spokojny_full_log.copy()
test_spokojny_full_log_scaled = test_spokojny_full_log.copy()

train_spokojny_full_log_scaled[columns_to_scale_full_log] = scaler_full_log.fit_transform(train_spokojny_full_log[columns_to_scale_full_log])
test_spokojny_full_log_scaled[columns_to_scale_full_log] = scaler_full_log.transform(test_spokojny_full_log[columns_to_scale_full_log])

df_full_log = pd.concat([train_spokojny_full_log_scaled, test_spokojny_full_log_scaled])

# --- Preprocessing dla skróconego zbioru danych ---

# 1. Uzupełnienie brakujących wartości
# df_short = df_short.interpolate(method="linear").fillna(method="ffill").fillna(method="bfill")

# 2. Kodowanie zmiennych cyklicznych (tylko sinus)
df_short["day_of_week_sin"] = np.sin(2 * np.pi * df_short["day_of_week"] / 7)
df_short["month_sin"] = np.sin(2 * np.pi * df_short["month"] / 12)
df_short["hour_sin"] = np.sin(2 * np.pi * df_short["hour"] / 24)
df_short = df_short.drop(columns=["day_of_week", "month", "hour"])

# Usuń zmienne _cos, jeśli istnieją
cos_columns = [col for col in df_short.columns if col.endswith("_cos")]
if cos_columns:
    df_short = df_short.drop(columns=cos_columns)

# 3. Kopia dla logarytmizacji
df_short_log = df_short.copy()
df_short_log["fixing_i_price_log"] = np.log(df_short_log["fixing_i_price"] + 1 + epsilon)

# 4. Standaryzacja danych (bez logarytmizacji)
train_spokojny_short = df_short[(df_short["timestamp"].dt.year >= 2016) & (df_short["timestamp"].dt.year <= 2018)]
test_spokojny_short = df_short[df_short["timestamp"].dt.year == 2019]

columns_to_scale_short = [col for col in df_short.columns if col not in ["timestamp", "is_holiday"]]

scaler_short = StandardScaler()
train_spokojny_short_scaled = train_spokojny_short.copy()
test_spokojny_short_scaled = test_spokojny_short.copy()

train_spokojny_short_scaled[columns_to_scale_short] = scaler_short.fit_transform(train_spokojny_short[columns_to_scale_short])
test_spokojny_short_scaled[columns_to_scale_short] = scaler_short.transform(test_spokojny_short_scaled[columns_to_scale_short])

df_short = pd.concat([train_spokojny_short_scaled, test_spokojny_short_scaled])

# 5. Standaryzacja danych (z logarytmizacją)
train_spokojny_short_log = df_short_log[(df_short_log["timestamp"].dt.year >= 2016) & (df_short_log["timestamp"].dt.year <= 2018)]
test_spokojny_short_log = df_short_log[df_short_log["timestamp"].dt.year == 2019]

columns_to_scale_short_log = [col for col in df_short_log.columns if col not in ["timestamp", "is_holiday"]]

scaler_short_log = StandardScaler()
train_spokojny_short_log_scaled = train_spokojny_short_log.copy()
test_spokojny_short_log_scaled = test_spokojny_short_log.copy()

train_spokojny_short_log_scaled[columns_to_scale_short_log] = scaler_short_log.fit_transform(train_spokojny_short_log[columns_to_scale_short_log])
test_spokojny_short_log_scaled[columns_to_scale_short_log] = scaler_short_log.transform(test_spokojny_short_log_scaled[columns_to_scale_short_log])

df_short_log = pd.concat([train_spokojny_short_log_scaled, test_spokojny_short_log_scaled])

# --- Regresja liniowa dla pełnego zbioru danych (bez logarytmizacji) ---

X_columns_full = [col for col in df_full.columns if col not in ["timestamp", "fixing_i_price"]]
train_spokojny_full = df_full[(df_full["timestamp"].dt.year >= 2016) & (df_full["timestamp"].dt.year <= 2018)]
test_spokojny_full = df_full[df_full["timestamp"].dt.year == 2019]

X_train_full = train_spokojny_full[X_columns_full]
y_train_full = train_spokojny_full["fixing_i_price"]
X_test_full = test_spokojny_full[X_columns_full]
y_test_full = test_spokojny_full["fixing_i_price"]

# Trening i ocena
linear_model_full = LinearRegression()
linear_model_full.fit(X_train_full, y_train_full)
linear_predictions_full = linear_model_full.predict(X_test_full)

# Odwrócenie standaryzacji
price_index = columns_to_scale_full.index("fixing_i_price")
y_test_full_original = scaler_full.mean_[price_index] + scaler_full.scale_[price_index] * y_test_full
linear_predictions_full_original = scaler_full.mean_[price_index] + scaler_full.scale_[price_index] * linear_predictions_full

# Obliczenie metryk na oryginalnej skali
linear_mae_full, linear_rmse_full, linear_mape_full, linear_smape_full, linear_r2_full = calculate_metrics(y_test_full_original, linear_predictions_full_original)

print("\nRegresja liniowa (pełny zbiór danych) - Okres stabilny (2019):")
print(f"MAE = {linear_mae_full:.2f}, RMSE = {linear_rmse_full:.2f}, MAPE = {linear_mape_full:.2f}%, sMAPE = {linear_smape_full:.2f}%, R^2 = {linear_r2_full:.4f}")

# --- Regresja liniowa dla skróconego zbioru danych (bez logarytmizacji) ---

X_columns_short = [col for col in df_short.columns if col not in ["timestamp", "fixing_i_price"]]
train_spokojny_short = df_short[(df_short["timestamp"].dt.year >= 2016) & (df_short["timestamp"].dt.year <= 2018)]
test_spokojny_short = df_short[df_short["timestamp"].dt.year == 2019]

X_train_short = train_spokojny_short[X_columns_short]
y_train_short = train_spokojny_short["fixing_i_price"]
X_test_short = test_spokojny_short[X_columns_short]
y_test_short = test_spokojny_short["fixing_i_price"]

# Trening i ocena
linear_model_short = LinearRegression()
linear_model_short.fit(X_train_short, y_train_short)
linear_predictions_short = linear_model_short.predict(X_test_short)

# Odwrócenie standaryzacji
price_index_short = columns_to_scale_short.index("fixing_i_price")
y_test_short_original = scaler_short.mean_[price_index_short] + scaler_short.scale_[price_index_short] * y_test_short
linear_predictions_short_original = scaler_short.mean_[price_index_short] + scaler_short.scale_[price_index_short] * linear_predictions_short

# Obliczenie metryk
linear_mae_short, linear_rmse_short, linear_mape_short, linear_smape_short, linear_r2_short = calculate_metrics(y_test_short_original, linear_predictions_short_original)

print("\nRegresja liniowa (skrócony zbiór danych) - Okres stabilny (2019):")
print(f"MAE = {linear_mae_short:.2f}, RMSE = {linear_rmse_short:.2f}, MAPE = {linear_mape_short:.2f}%, sMAPE = {linear_smape_short:.2f}%, R^2 = {linear_r2_short:.4f}")

# --- Regresja Ridge dla pełnego zbioru danych (bez logarytmizacji) ---

tscv = TimeSeriesSplit(n_splits=5)
ridge_params = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=tscv, scoring="neg_mean_squared_error")

# Trening i ocena
ridge_grid.fit(X_train_full, y_train_full)
best_ridge_full = ridge_grid.best_estimator_
ridge_predictions_full = best_ridge_full.predict(X_test_full)

# Odwrócenie standaryzacji
ridge_predictions_full_original = scaler_full.mean_[price_index] + scaler_full.scale_[price_index] * ridge_predictions_full

# Obliczenie metryk
ridge_mae_full, ridge_rmse_full, ridge_mape_full, ridge_smape_full, ridge_r2_full = calculate_metrics(y_test_full_original, ridge_predictions_full_original)

print("\nRegresja Ridge (pełny zbiór danych) - Okres stabilny (2019):")
print(f"MAE = {ridge_mae_full:.2f}, RMSE = {ridge_rmse_full:.2f}, MAPE = {ridge_mape_full:.2f}%, sMAPE = {ridge_smape_full:.2f}%, R^2 = {ridge_r2_full:.4f}")
print(f"Najlepsze alpha: {ridge_grid.best_params_['alpha']}")

# --- Regresja Ridge dla skróconego zbioru danych (bez logarytmizacji) ---

ridge_grid.fit(X_train_short, y_train_short)
best_ridge_short = ridge_grid.best_estimator_
ridge_predictions_short = best_ridge_short.predict(X_test_short)

# Odwrócenie standaryzacji
ridge_predictions_short_original = scaler_short.mean_[price_index_short] + scaler_short.scale_[price_index_short] * ridge_predictions_short

# Obliczenie metryk
ridge_mae_short, ridge_rmse_short, ridge_mape_short, ridge_smape_short, ridge_r2_short = calculate_metrics(y_test_short_original, ridge_predictions_short_original)

print("\nRegresja Ridge (skrócony zbiór danych) - Okres stabilny (2019):")
print(f"MAE = {ridge_mae_short:.2f}, RMSE = {ridge_rmse_short:.2f}, MAPE = {ridge_mape_short:.2f}%, sMAPE = {ridge_smape_short:.2f}%, R^2 = {ridge_r2_short:.4f}")
print(f"Najlepsze alpha: {ridge_grid.best_params_['alpha']}")

# --- Regresja liniowa dla pełnego zbioru danych (z logarytmizacją) ---

X_columns_full_log = [col for col in df_full_log.columns if col not in ["timestamp", "fixing_i_price", "fixing_i_price_log"]]
train_spokojny_full_log = df_full_log[(df_full_log["timestamp"].dt.year >= 2016) & (df_full_log["timestamp"].dt.year <= 2018)]
test_spokojny_full_log = df_full_log[df_full_log["timestamp"].dt.year == 2019]

X_train_full_log = train_spokojny_full_log[X_columns_full_log]
y_train_full_log = train_spokojny_full_log["fixing_i_price_log"]
X_test_full_log = test_spokojny_full_log[X_columns_full_log]
y_test_full_log = test_spokojny_full_log["fixing_i_price_log"]
y_test_full_log_original = test_spokojny_full_log["fixing_i_price"]

# Trening i ocena
linear_model_full_log = LinearRegression()
linear_model_full_log.fit(X_train_full_log, y_train_full_log)
linear_predictions_log_full = linear_model_full_log.predict(X_test_full_log)

# Odwrócenie logarytmizacji i standaryzacji
log_price_index = columns_to_scale_full_log.index("fixing_i_price_log")
y_test_log_scaled = scaler_full_log.mean_[log_price_index] + scaler_full_log.scale_[log_price_index] * y_test_full_log
predictions_log_scaled = scaler_full_log.mean_[log_price_index] + scaler_full_log.scale_[log_price_index] * linear_predictions_log_full

linear_predictions_full_log = np.exp(predictions_log_scaled) - 1 - epsilon
y_test_full_restored = np.exp(y_test_log_scaled) - 1 - epsilon

# Obliczenie metryk na oryginalnej skali
linear_mae_full_log, linear_rmse_full_log, linear_mape_full_log, linear_smape_full_log, linear_r2_full_log = calculate_metrics(y_test_full_restored, linear_predictions_full_log)

print("\nRegresja liniowa z logarytmizacją (pełny zbiór danych) - Okres stabilny (2019):")
print(f"MAE = {linear_mae_full_log:.2f}, RMSE = {linear_rmse_full_log:.2f}, MAPE = {linear_mape_full_log:.2f}%, sMAPE = {linear_smape_full_log:.2f}%, R^2 = {linear_r2_full_log:.4f}")

# --- Regresja liniowa dla skróconego zbioru danych (z logarytmizacją) ---

X_columns_short_log = [col for col in df_short_log.columns if col not in ["timestamp", "fixing_i_price", "fixing_i_price_log"]]
train_spokojny_short_log = df_short_log[(df_short_log["timestamp"].dt.year >= 2016) & (df_short_log["timestamp"].dt.year <= 2018)]
test_spokojny_short_log = df_short_log[df_short_log["timestamp"].dt.year == 2019]

X_train_short_log = train_spokojny_short_log[X_columns_short_log]
y_train_short_log = train_spokojny_short_log["fixing_i_price_log"]
X_test_short_log = test_spokojny_short_log[X_columns_short_log]
y_test_short_log = test_spokojny_short_log["fixing_i_price_log"]
y_test_short_log_original = test_spokojny_short_log["fixing_i_price"]

# Trening i ocena
linear_model_short_log = LinearRegression()
linear_model_short_log.fit(X_train_short_log, y_train_short_log)
linear_predictions_log_short = linear_model_short_log.predict(X_test_short_log)

# Odwrócenie logarytmizacji i standaryzacji
log_price_index_short = columns_to_scale_short_log.index("fixing_i_price_log")
y_test_log_scaled_short = scaler_short_log.mean_[log_price_index_short] + scaler_short_log.scale_[log_price_index_short] * y_test_short_log
predictions_log_scaled_short = scaler_short_log.mean_[log_price_index_short] + scaler_short_log.scale_[log_price_index_short] * linear_predictions_log_short

linear_predictions_short_log = np.exp(predictions_log_scaled_short) - 1 - epsilon
y_test_short_restored = np.exp(y_test_log_scaled_short) - 1 - epsilon

# Obliczenie metryk
linear_mae_short_log, linear_rmse_short_log, linear_mape_short_log, linear_smape_short_log, linear_r2_short_log = calculate_metrics(y_test_short_restored, linear_predictions_short_log)

print("\nRegresja liniowa z logarytmizacją (skrócony zbiór danych) - Okres stabilny (2019):")
print(f"MAE = {linear_mae_short_log:.2f}, RMSE = {linear_rmse_short_log:.2f}, MAPE = {linear_mape_short_log:.2f}%, sMAPE = {linear_smape_short_log:.2f}%, R^2 = {linear_r2_short_log:.4f}")

# --- Regresja Ridge dla pełnego zbioru danych (z logarytmizacją) ---

tscv = TimeSeriesSplit(n_splits=5)
ridge_params = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=tscv, scoring="neg_mean_squared_error")

# Trening i ocena
ridge_grid.fit(X_train_full_log, y_train_full_log)
best_ridge_full_log = ridge_grid.best_estimator_
ridge_predictions_log_full = best_ridge_full_log.predict(X_test_full_log)

# Odwrócenie logarytmizacji i standaryzacji
ridge_predictions_log_scaled_full = scaler_full_log.mean_[log_price_index] + scaler_full_log.scale_[log_price_index] * ridge_predictions_log_full
ridge_predictions_full_log = np.exp(ridge_predictions_log_scaled_full) - 1 - epsilon

# Obliczenie metryk
ridge_mae_full_log, ridge_rmse_full_log, ridge_mape_full_log, ridge_smape_full_log, ridge_r2_full_log = calculate_metrics(y_test_full_restored, ridge_predictions_full_log)

print("\nRegresja Ridge z logarytmizacją (pełny zbiór danych) - Okres stabilny (2019):")
print(f"MAE = {ridge_mae_full_log:.2f}, RMSE = {ridge_rmse_full_log:.2f}, MAPE = {ridge_mape_full_log:.2f}%, sMAPE = {ridge_smape_full_log:.2f}%, R^2 = {ridge_r2_full_log:.4f}")
print(f"Najlepsze alpha: {ridge_grid.best_params_['alpha']}")

# --- Regresja Ridge dla skróconego zbioru danych (z logarytmizacją) ---

ridge_grid.fit(X_train_short_log, y_train_short_log)
best_ridge_short_log = ridge_grid.best_estimator_
ridge_predictions_log_short = best_ridge_short_log.predict(X_test_short_log)

# Odwrócenie logarytmizacji i standaryzacji
ridge_predictions_log_scaled_short = scaler_short_log.mean_[log_price_index_short] + scaler_short_log.scale_[log_price_index_short] * ridge_predictions_log_short
ridge_predictions_short_log = np.exp(ridge_predictions_log_scaled_short) - 1 - epsilon

# Obliczenie metryk
ridge_mae_short_log, ridge_rmse_short_log, ridge_mape_short_log, ridge_smape_short_log, ridge_r2_short_log = calculate_metrics(y_test_short_restored, ridge_predictions_short_log)

print("\nRegresja Ridge z logarytmizacją (skrócony zbiór danych) - Okres stabilny (2019):")
print(f"MAE = {ridge_mae_short_log:.2f}, RMSE = {ridge_rmse_short_log:.2f}, MAPE = {ridge_mape_short_log:.2f}%, sMAPE = {ridge_smape_short_log:.2f}%, R^2 = {ridge_r2_short_log:.4f}")
print(f"Najlepsze alpha: {ridge_grid.best_params_['alpha']}")

# Histogram reszt dla regresji Ridge (pełny zbiór)
residuals_ridge_full = y_test_full_original - ridge_predictions_full_original
plt.figure(figsize=(6, 4))
plt.hist(residuals_ridge_full, bins=50, color="red", alpha=0.7)
plt.title("Histogram reszt - Regresja Ridge (pełny zbiór)", fontsize=12)
plt.xlabel("Reszty", fontsize=10)
plt.ylabel("Częstość", fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig("../../plots/predicts/residuals_histogram_Ridge_full_stable_period.png", dpi=300)
plt.close()

# Wykres predykcji vs. wartości rzeczywistych
plt.figure(figsize=(12, 4))
plt.plot(test_spokojny_full["timestamp"], y_test_full_original, label="Rzeczywiste", color="blue", alpha=0.7)
plt.plot(test_spokojny_full["timestamp"], ridge_predictions_full_original, label="Regresja Ridge", color="red", alpha=0.7)
plt.title("Okres stabilny (2019)", fontsize=14)
plt.xlabel("Czas", fontsize=12)
plt.ylabel("Cena energii [PLN/MWh]", fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("../../plots/predicts/ridge_predictions_full_stable_period.png", dpi=300)
plt.close()

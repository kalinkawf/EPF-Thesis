import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import os

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

# Rozpoczęcie mierzenia czasu
start_time = time.time()

# --- Wczytanie danych ---

df_full = pd.read_csv("../../data/database.csv")
df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])

df_short = pd.read_csv("../../data/short_database.csv")
df_short["timestamp"] = pd.to_datetime(df_short["timestamp"])

# --- Preprocessing dla pełnego zbioru danych ---

# 1. Uzupełnienie brakujących wartości (odkomentuj, jeśli potrzebne)
# df_full = df_full.interpolate(method="linear").fillna(method="ffill").fillna(method="bfill")

# 2. Kodowanie zmiennych cyklicznych (tylko sinus)
df_full["day_of_week_sin"] = np.sin(2 * np.pi * df_full["day_of_week"] / 7)
df_full["month_sin"] = np.sin(2 * np.pi * df_full["month"] / 12)
df_full["hour_sin"] = np.sin(2 * np.pi * df_full["hour"] / 24)
df_full = df_full.drop(columns=["day_of_week", "month", "hour"])

# 3. Standaryzacja danych
train_spokojny_full = df_full[(df_full["timestamp"].dt.year >= 2016) & (df_full["timestamp"].dt.year <= 2018)]
test_spokojny_full = df_full[df_full["timestamp"].dt.year == 2019]

# Definicja kolumn do standaryzacji (wykluczamy fixing_i_price i timestamp)
columns_to_scale_full = [col for col in df_full.columns if col not in ["timestamp", "fixing_i_price", "is_holiday"]]

scaler_full = StandardScaler()
train_spokojny_full_scaled = train_spokojny_full.copy()
test_spokojny_full_scaled = test_spokojny_full.copy()

# Standaryzacja tylko cech, nie fixing_i_price
train_spokojny_full_scaled[columns_to_scale_full] = scaler_full.fit_transform(train_spokojny_full[columns_to_scale_full])
test_spokojny_full_scaled[columns_to_scale_full] = scaler_full.transform(test_spokojny_full[columns_to_scale_full])

# Zachowujemy fixing_i_price jako oryginalne wartości
y_test_full_original = test_spokojny_full["fixing_i_price"].values
y_train_full_original = train_spokojny_full["fixing_i_price"].values

df_full = pd.concat([train_spokojny_full_scaled, test_spokojny_full_scaled])

# --- Preprocessing dla skróconego zbioru danych ---

# 1. Uzupełnienie brakujących wartości (odkomentuj, jeśli potrzebne)
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

# 3. Standaryzacja danych
train_spokojny_short = df_short[(df_short["timestamp"].dt.year >= 2016) & (df_short["timestamp"].dt.year <= 2018)]
test_spokojny_short = df_short[df_short["timestamp"].dt.year == 2019]

columns_to_scale_short = [col for col in df_short.columns if col not in ["timestamp", "fixing_i_price", "is_holiday"]]

scaler_short = StandardScaler()
train_spokojny_short_scaled = train_spokojny_short.copy()
test_spokojny_short_scaled = test_spokojny_short.copy()

# Standaryzacja tylko cech, nie fixing_i_price
train_spokojny_short_scaled[columns_to_scale_short] = scaler_short.fit_transform(train_spokojny_short[columns_to_scale_short])
test_spokojny_short_scaled[columns_to_scale_short] = scaler_short.transform(test_spokojny_short[columns_to_scale_short])

# Zachowujemy fixing_i_price jako oryginalne wartości
y_test_short_original = test_spokojny_short["fixing_i_price"].values
y_train_short_original = train_spokojny_short["fixing_i_price"].values

df_short = pd.concat([train_spokojny_short_scaled, test_spokojny_short_scaled])

# --- Regresja liniowa dla pełnego zbioru danych ---

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

# Ponieważ fixing_i_price nie był standaryzowany, predykcje są już w oryginalnej skali
linear_predictions_full_original = linear_predictions_full

# Obliczenie metryk (tylko do wyświetlenia)
linear_mae_full, linear_rmse_full, linear_mape_full, linear_smape_full, linear_r2_full = calculate_metrics(y_test_full_original, linear_predictions_full_original)

print("\nRegresja liniowa (pełny zbiór danych) - Okres stabilny (2019):")
print(f"MAE = {linear_mae_full:.2f}, RMSE = {linear_rmse_full:.2f}, MAPE = {linear_mape_full:.2f}%, sMAPE = {linear_smape_full:.2f}%, R^2 = {linear_r2_full:.4f}")

# Zapis predykcji i wartości rzeczywistych
predictions_df_full_linear = pd.DataFrame({
    'timestamp': test_spokojny_full['timestamp'],
    'fixing_i_price': y_test_full_original,
    'prediction': linear_predictions_full_original
})
os.makedirs("../../results", exist_ok=True)
predictions_df_full_linear.to_csv("../../results/predictions_full_linear_stable.csv", index=False)

# --- Regresja liniowa dla skróconego zbioru danych ---

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

# Ponieważ fixing_i_price nie był standaryzowany, predykcje są już w oryginalnej skali
linear_predictions_short_original = linear_predictions_short

# Obliczenie metryk (tylko do wyświetlenia)
linear_mae_short, linear_rmse_short, linear_mape_short, linear_smape_short, linear_r2_short = calculate_metrics(y_test_short_original, linear_predictions_short_original)

print("\nRegresja liniowa (skrócony zbiór danych) - Okres stabilny (2019):")
print(f"MAE = {linear_mae_short:.2f}, RMSE = {linear_rmse_short:.2f}, MAPE = {linear_mape_short:.2f}%, sMAPE = {linear_smape_short:.2f}%, R^2 = {linear_r2_short:.4f}")

# Zapis predykcji i wartości rzeczywistych
predictions_df_short_linear = pd.DataFrame({
    'timestamp': test_spokojny_short['timestamp'],
    'fixing_i_price': y_test_short_original,
    'prediction': linear_predictions_short_original
})
predictions_df_short_linear.to_csv("../../results/predictions_short_linear_stable.csv", index=False)

# --- Regresja Ridge dla pełnego zbioru danych ---

tscv = TimeSeriesSplit(n_splits=5)
ridge_params = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=tscv, scoring="neg_mean_squared_error")

# Trening i ocena
ridge_grid.fit(X_train_full, y_train_full)
best_ridge_full = ridge_grid.best_estimator_
ridge_predictions_full = best_ridge_full.predict(X_test_full)

# Ponieważ fixing_i_price nie był standaryzowany, predykcje są już w oryginalnej skali
ridge_predictions_full_original = ridge_predictions_full

# Obliczenie metryk (tylko do wyświetlenia)
ridge_mae_full, ridge_rmse_full, ridge_mape_full, ridge_smape_full, ridge_r2_full = calculate_metrics(y_test_full_original, ridge_predictions_full_original)

print("\nRegresja Ridge (pełny zbiór danych) - Okres stabilny (2019):")
print(f"MAE = {ridge_mae_full:.2f}, RMSE = {ridge_rmse_full:.2f}, MAPE = {ridge_mape_full:.2f}%, sMAPE = {ridge_smape_full:.2f}%, R^2 = {ridge_r2_full:.4f}")
print(f"Najlepsze alpha: {ridge_grid.best_params_['alpha']}")

# Zapis predykcji i wartości rzeczywistych
predictions_df_full_ridge = pd.DataFrame({
    'timestamp': test_spokojny_full['timestamp'],
    'fixing_i_price': y_test_full_original,
    'prediction': ridge_predictions_full_original
})
predictions_df_full_ridge.to_csv("../../results/predictions_full_ridge_stable.csv", index=False)

# --- Regresja Ridge dla skróconego zbioru danych ---

ridge_grid.fit(X_train_short, y_train_short)
best_ridge_short = ridge_grid.best_estimator_
ridge_predictions_short = best_ridge_short.predict(X_test_short)

# Ponieważ fixing_i_price nie był standaryzowany, predykcje są już w oryginalnej skali
ridge_predictions_short_original = ridge_predictions_short

# Obliczenie metryk (tylko do wyświetlenia)
ridge_mae_short, ridge_rmse_short, ridge_mape_short, ridge_smape_short, ridge_r2_short = calculate_metrics(y_test_short_original, ridge_predictions_short_original)

print("\nRegresja Ridge (skrócony zbiór danych) - Okres stabilny (2019):")
print(f"MAE = {ridge_mae_short:.2f}, RMSE = {ridge_rmse_short:.2f}, MAPE = {ridge_mape_short:.2f}%, sMAPE = {ridge_smape_short:.2f}%, R^2 = {ridge_r2_short:.4f}")
print(f"Najlepsze alpha: {ridge_grid.best_params_['alpha']}")

# Zapis predykcji i wartości rzeczywistych
predictions_df_short_ridge = pd.DataFrame({
    'timestamp': test_spokojny_short['timestamp'],
    'fixing_i_price': y_test_short_original,
    'prediction': ridge_predictions_short_original
})
predictions_df_short_ridge.to_csv("../../results/predictions_short_ridge_stable.csv", index=False)

# --- Histogram reszt dla regresji Ridge (pełny zbiór) ---
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

# --- Wykres predykcji vs. wartości rzeczywistych ---
plt.figure(figsize=(12, 4))
plt.plot(test_spokojny_full["timestamp"], y_test_full_original, label="Rzeczywiste", color="blue", alpha=0.7)
plt.plot(test_spokojny_full["timestamp"], ridge_predictions_full_original, label="Regresja Ridge", color="red", alpha=0.7, linestyle="--")
plt.title("Okres stabilny (2019)", fontsize=14)
plt.xlabel("Czas", fontsize=12)
plt.ylabel("Cena energii [PLN/MWh]", fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("../../plots/predicts/ridge_predictions_full_stable_period.png", dpi=300)
plt.close()

# Wykres predykcji oraz rzeczywistych wartości od 01.06 do 31.06
june_mask = (test_spokojny_full["timestamp"] >= "2019-06-01") & (test_spokojny_full["timestamp"] <= "2019-06-30")
plt.figure(figsize=(12, 4))
plt.plot(test_spokojny_full.loc[june_mask, "timestamp"], y_test_full_original[june_mask], label="Rzeczywiste", color="blue", alpha=0.7)
plt.plot(test_spokojny_full.loc[june_mask, "timestamp"], ridge_predictions_full_original[june_mask], label="Regresja Ridge", color="red", alpha=0.7, linestyle="--")
plt.title("Okres stabilny (Czerwiec 2019)", fontsize=14)
plt.xlabel("Czas", fontsize=12)
plt.ylabel("Cena energii [PLN/MWh]", fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("../../plots/predicts/ridge_predictions_full_june_2019.png", dpi=300)
plt.close()

# Wykres predykcji oraz rzeczywistych wartości od 01.01 do 31.01
january_mask = (test_spokojny_full["timestamp"] >= "2019-01-01") & (test_spokojny_full["timestamp"] <= "2019-01-31")
plt.figure(figsize=(12, 4))
plt.plot(test_spokojny_full.loc[january_mask, "timestamp"], y_test_full_original[january_mask], label="Rzeczywiste", color="blue", alpha=0.7)
plt.plot(test_spokojny_full.loc[january_mask, "timestamp"], ridge_predictions_full_original[january_mask], label="Regresja Ridge", color="red", alpha=0.7, linestyle="--")
plt.title("Okres stabilny (Styczeń 2019)", fontsize=14)
plt.xlabel("Czas", fontsize=12)
plt.ylabel("Cena energii [PLN/MWh]", fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("../../plots/predicts/ridge_predictions_full_january_2019.png", dpi=300)
plt.close()

# Wykres predykcji oraz rzeczywistych wartości od 01.01 do 31.03 (Q1)
q1_mask = (test_spokojny_full["timestamp"] >= "2019-01-01") & (test_spokojny_full["timestamp"] <= "2019-03-31")
plt.figure(figsize=(12, 4))
plt.plot(test_spokojny_full.loc[q1_mask, "timestamp"], y_test_full_original[q1_mask], label="Rzeczywiste", color="blue", alpha=0.7)
plt.plot(test_spokojny_full.loc[q1_mask, "timestamp"], ridge_predictions_full_original[q1_mask], label="Regresja Ridge", color="red", alpha=0.7, linestyle="--")
plt.title("Okres stabilny (Q1 2019)", fontsize=14)
plt.xlabel("Czas", fontsize=12)
plt.ylabel("Cena energii [PLN/MWh]", fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("../../plots/predicts/ridge_predictions_full_q1_2019.png", dpi=300)
plt.close()

# --- Wykres top 50 błędów (Regresja Ridge, pełny zbiór) ---
residuals_abs = np.abs(residuals_ridge_full)
top_50_indices = np.argsort(residuals_abs)[-50:]  # Indeksy 20 największych błędów
top_50_actual = y_test_full_original[top_50_indices]
top_50_pred = ridge_predictions_full_original[top_50_indices]
top_50_timestamps = test_spokojny_full.iloc[top_50_indices]["timestamp"].values

# Wyprintowanie timestampów największych błędów
print("\nTimestampy największych błędów:")
print(top_50_timestamps)

plt.figure(figsize=(8, 6))
plt.scatter(top_50_pred, top_50_actual, color="red", alpha=0.7)
plt.plot([min(top_50_actual.min(), top_50_pred.min()), max(top_50_actual.max(), top_50_pred.max())],
         [min(top_50_actual.min(), top_50_pred.min()), max(top_50_actual.max(), top_50_pred.max())],
         color="black", linestyle="--", label="Linia idealna")
plt.title("Top 50 błędów: Predykcje vs Rzeczywiste (Ridge, pełny zbiór)", fontsize=12)
plt.xlabel("Predykcje [PLN/MWh]", fontsize=10)
plt.ylabel("Rzeczywiste wartości [PLN/MWh]", fontsize=10)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("../../plots/predicts/top_50_errors_Ridge_full_stable_period.png", dpi=300)
plt.close()

# --- Wykres wszystkich błędów vs predykcji z linią przerywaną ---
residuals_all = y_test_full_original - ridge_predictions_full_original
plt.figure(figsize=(8, 6))
plt.scatter(ridge_predictions_full_original, residuals_all, color="orange", alpha=0.5, label="Błędy")
plt.axhline(y=0, color="black", linestyle="--", label="Linia zerowa")
plt.title("Błędy vs Predykcje (Ridge, pełny zbiór)", fontsize=12)
plt.xlabel("Predykcje [PLN/MWh]", fontsize=10)
plt.ylabel("Błędy (Rzeczywiste - Predykcje) [PLN/MWh]", fontsize=10)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("../../plots/predicts/errors_vs_predictions_Ridge_full_stable_period.png", dpi=300)
plt.close()

# --- Analiza najważniejszych czynników (Regresja Ridge, pełny zbiór) ---

# Współczynniki modelu Ridge
feature_importance = pd.DataFrame({
    'Feature': X_columns_full,
    'Coefficient': best_ridge_full.coef_
})
feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values(by='Abs_Coefficient', ascending=False)

print("\nNajważniejsze czynniki wpływające na cenę (Ridge, pełny zbiór):")
print(feature_importance[['Feature', 'Coefficient']].head(10))

# Wykres 10 najważniejszych czynników
top_10_features = feature_importance.head(10)
plt.figure(figsize=(10, 6))
plt.barh(top_10_features['Feature'], top_10_features['Coefficient'], color="teal")
plt.title("Top 10 najważniejszych czynników (Ridge, pełny zbiór)", fontsize=12)
plt.xlabel("Współczynnik", fontsize=10)
plt.ylabel("Cecha", fontsize=10)
plt.grid(True, axis="x")
plt.tight_layout()
plt.savefig("../../plots/predicts/feature_importance_Ridge_full_stable_period.png", dpi=300)
plt.close()

# --- Czas wykonania ---
execution_time = time.time() - start_time
print(f"\nCzas wykonania: {execution_time:.2f} sekund")
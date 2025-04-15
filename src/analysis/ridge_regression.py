# Importowanie bibliotek
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Wczytanie danych
df_full = pd.read_csv("../../data/database.csv")
df_short = pd.read_csv("../../data/short_database.csv")

# Konwersja timestamp na format datetime
df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])
df_short["timestamp"] = pd.to_datetime(df_short["timestamp"])

# Konwersja zmiennych logicznych na numeryczne
df_full["is_holiday"] = df_full["is_holiday"].astype(int)
df_short["is_holiday"] = df_short["is_holiday"].astype(int)

# Lista zmiennych objaśniających dla pełnego zestawu danych
features_full = [
    "temp_waw", "wind_speed_waw", "cloud_cover_waw", "solar_radiation_waw",
    "temp_ksz", "wind_speed_ksz", "cloud_cover_ksz", "solar_radiation_ksz",
    "temp_krk", "wind_speed_krk", "cloud_cover_krk", "solar_radiation_krk",
    "temp_bab", "wind_speed_bab", "cloud_cover_bab", "solar_radiation_bab",
    "power_loss", "Network_loss",
    "Niemcy Bilans", "Czechy Bilans", "Litwa Bilans", "Słowacja Bilans", "Szwecja Bilans", "Ukraina Bilans",
    "hard_coal", "coal-derived", "lignite", "gas", "oil", "biomass", "wind_onshore", "solar",
    "fixing_i_volume", "Load", "gas_price", "coal_pscmi1_pln_per_gj", "co2_price",
    "pln_usd", "brent_price", "day_of_week", "month", "hour", "fixing_i_price_lag24", "fixing_i_price_lag168", "is_holiday"
]

# Lista zmiennych objaśniających dla skróconego zestawu danych
features_short = [
    "fixing_i_price_lag24", "fixing_i_price_lag168",
    "gas_price", "co2_price", "brent_price", "pln_usd", "coal_pscmi1_pln_per_gj",
    "power_loss", "fixing_i_volume", "solar", "gas", "oil", "Load",
    "avg_temp", "avg_wind_speed", "avg_solar_radiation",
    "hour", "month", "is_holiday", "wind_onshore", "day_of_week"
]

target = "fixing_i_price"

# Usunięcie brakujących wartości
df_full = df_full[features_full + [target, "timestamp"]].dropna()
df_short = df_short[features_short + [target, "timestamp"]].dropna()

# Funkcja do obliczania metryk (z poprawką na MAPE)
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    smape = np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100
    naive_forecast = y_true.shift(1).dropna()
    y_true_mase = y_true[1:]
    y_pred_mase = y_pred[1:]
    naive_error = np.mean(np.abs(y_true_mase - naive_forecast))
    mase = np.mean(np.abs(y_true_mase - y_pred_mase)) / naive_error
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MSE": mse, "MAPE": mape, "sMAPE": smape, "MASE": mase, "R2": r2}

# Funkcja do trenowania i oceny modelu z GridSearchCV
def train_and_evaluate(features, period_name, train_data, test_data):
    scaler = StandardScaler()
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    # Skalowanie danych
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Definicja modelu Ridge i siatki parametrów
    ridge = Ridge()
    param_grid = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    
    # GridSearchCV do znalezienia optymalnego alpha
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Najlepszy model
    best_ridge = grid_search.best_estimator_
    print(f"Najlepsze alpha dla {period_name}: {grid_search.best_params_['alpha']}")

    # Prognozowanie
    y_pred = best_ridge.predict(X_test_scaled)

    # Obliczanie metryk
    metrics = calculate_metrics(y_test, y_pred)
    return metrics, y_test, y_pred

# Przygotowanie danych dla obu okresów (pełny zestaw)
train_spokojny_full = df_full[(df_full["timestamp"].dt.year >= 2016) & (df_full["timestamp"].dt.year <= 2018)]
test_spokojny_full = df_full[df_full["timestamp"].dt.year == 2019]
train_niespokojny_full = df_full[(df_full["timestamp"].dt.year >= 2019) & (df_full["timestamp"].dt.year <= 2022)]
test_niespokojny_full = df_full[df_full["timestamp"].dt.year == 2023]

# Przygotowanie danych dla obu okresów (skrócony zestaw)
train_spokojny_short = df_short[(df_short["timestamp"].dt.year >= 2016) & (df_short["timestamp"].dt.year <= 2018)]
test_spokojny_short = df_short[df_short["timestamp"].dt.year == 2019]
train_niespokojny_short = df_short[(df_short["timestamp"].dt.year >= 2019) & (df_short["timestamp"].dt.year <= 2022)]
test_niespokojny_short = df_short[df_short["timestamp"].dt.year == 2023]

# Trenowanie i ocena dla pełnego zestawu danych
metrics_spokojny_full, y_test_spokojny_full, y_pred_spokojny_full = train_and_evaluate(
    features_full, "Spokojny (2019, pełny)", train_spokojny_full, test_spokojny_full
)
metrics_niespokojny_full, y_test_niespokojny_full, y_pred_niespokojny_full = train_and_evaluate(
    features_full, "Niespokojny (2023, pełny)", train_niespokojny_full, test_niespokojny_full
)

# Trenowanie i ocena dla skróconego zestawu danych
metrics_spokojny_short, y_test_spokojny_short, y_pred_spokojny_short = train_and_evaluate(
    features_short, "Spokojny (2019, skrócony)", train_spokojny_short, test_spokojny_short
)
metrics_niespokojny_short, y_test_niespokojny_short, y_pred_niespokojny_short = train_and_evaluate(
    features_short, "Niespokojny (2023, skrócony)", train_niespokojny_short, test_niespokojny_short
)

# Tworzenie DataFrame z wynikami
results_df = pd.DataFrame({
    "Metryka": ["MAE (PLN/MWh)", "RMSE (PLN/MWh)", "MSE (PLN/MWh)^2", "MAPE (%)", "sMAPE (%)", "MASE", "R2"],
    "Spokojny (2019, pełny)": [metrics_spokojny_full["MAE"], metrics_spokojny_full["RMSE"], metrics_spokojny_full["MSE"], 
                               metrics_spokojny_full["MAPE"], metrics_spokojny_full["sMAPE"], metrics_spokojny_full["MASE"], 
                               metrics_spokojny_full["R2"]],
    "Spokojny (2019, skrócony)": [metrics_spokojny_short["MAE"], metrics_spokojny_short["RMSE"], metrics_spokojny_short["MSE"], 
                                  metrics_spokojny_short["MAPE"], metrics_spokojny_short["sMAPE"], metrics_spokojny_short["MASE"], 
                                  metrics_spokojny_short["R2"]],
    "Niespokojny (2023, pełny)": [metrics_niespokojny_full["MAE"], metrics_niespokojny_full["RMSE"], metrics_niespokojny_full["MSE"], 
                                  metrics_niespokojny_full["MAPE"], metrics_niespokojny_full["sMAPE"], metrics_niespokojny_full["MASE"], 
                                  metrics_niespokojny_full["R2"]],
    "Niespokojny (2023, skrócony)": [metrics_niespokojny_short["MAE"], metrics_niespokojny_short["RMSE"], metrics_niespokojny_short["MSE"], 
                                     metrics_niespokojny_short["MAPE"], metrics_niespokojny_short["sMAPE"], metrics_niespokojny_short["MASE"], 
                                     metrics_niespokojny_short["R2"]]
})

# Zaokrąglenie wyników
results_df.iloc[:, 1:] = results_df.iloc[:, 1:].round(2)

# Wyświetlenie wyników
print("Wyniki dla Regresji Ridge z StandardScaler (porównanie pełnego i skróconego zestawu danych):")
print(results_df)

# Zapisywanie wyników w formacie LaTeX
latex_table = results_df.to_latex(index=False, float_format="%.2f", 
                                  caption="Wyniki modelu Regresji Ridge z StandardScaler dla pełnego i skróconego zestawu danych (zoptymalizowane alpha).",
                                  label="tab:wyniki_ridge_scaled_comparison")
print("\nTabela w formacie LaTeX:")
print(latex_table)

# Wykres prognoz vs rzeczywiste wartości dla okresu niespokojnego (2023, pełny zestaw)
plt.figure(figsize=(12, 6))
plt.plot(test_niespokojny_full["timestamp"], y_test_niespokojny_full, label='Rzeczywiste ceny', color='blue')
plt.plot(test_niespokojny_full["timestamp"], y_pred_niespokojny_full, label='Prognozy Regresji Ridge (pełny zestaw)', color='red', linestyle='--')
plt.title('Prognozy cen energii (Regresja Ridge z StandardScaler, okres niespokojny 2023, pełny zestaw)')
plt.xlabel('Data')
plt.ylabel('Cena (PLN/MWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../../plots/prognozy_ridge_scaled_niespokojny_full.png", dpi=300)
plt.close()

# Wykres prognoz vs rzeczywiste wartości dla okresu niespokojnego (2023, skrócony zestaw)
plt.figure(figsize=(12, 6))
plt.plot(test_niespokojny_short["timestamp"], y_test_niespokojny_short, label='Rzeczywiste ceny', color='blue')
plt.plot(test_niespokojny_short["timestamp"], y_pred_niespokojny_short, label='Prognozy Regresji Ridge (skrócony zestaw)', color='red', linestyle='--')
plt.title('Prognozy cen energii (Regresja Ridge z StandardScaler, okres niespokojny 2023, skrócony zestaw)')
plt.xlabel('Data')
plt.ylabel('Cena (PLN/MWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../../plots/prognozy_ridge_scaled_niespokojny_short.png", dpi=300)
plt.close()

print("Wykresy zapisane w ../../plots/prognozy_ridge_scaled_niespokojny_full.png i ../../plots/prognozy_ridge_scaled_niespokojny_short.png")
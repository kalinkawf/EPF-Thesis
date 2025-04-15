# Importowanie bibliotek
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
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

# Przesunięcie danych, aby wszystkie wartości były dodatnie
min_value_full = df_full[target].min()
min_value_short = df_short[target].min()
shift_value = min(min_value_full, min_value_short)
if shift_value < 0:
    shift_value = abs(shift_value) + 1
else:
    shift_value = 1

print(f"Przesunięcie danych o: {shift_value}")
df_full[target] = df_full[target] + shift_value
df_short[target] = df_short[target] + shift_value

# Zastąpienie ewentualnych wartości ujemnych lub zerowych małą wartością dodatnią
df_full[target] = df_full[target].clip(lower=1e-5)
df_short[target] = df_short[target].clip(lower=1e-5)

# Transformacja logarytmiczna zmiennej celu
df_full[target] = np.log(df_full[target])
df_short[target] = np.log(df_short[target])

# Funkcja do obliczania metryk
def calculate_metrics(y_true, y_pred, shift_value):
    # Odwrócenie transformacji logarytmicznej i przesunięcia
    y_true = np.exp(y_true) - shift_value
    y_pred = np.exp(y_pred) - shift_value

    # Upewniamy się, że y_true i y_pred są pandas.Series
    if isinstance(y_true, np.ndarray):
        y_true = pd.Series(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    
    # Obliczanie MAPE z większym epsilon
    epsilon = 1e-5
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    # Obliczanie sMAPE z zabezpieczeniem przed zerowym mianownikiem
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
    smape = np.mean(np.abs(y_true - y_pred) / denominator) * 100

    # Obliczanie MASE z zabezpieczeniem przed zerowym mianownikiem
    naive_forecast = y_true.shift(1).dropna()
    y_true_mase = y_true[1:]
    y_pred_mase = y_pred[1:]
    naive_error = np.mean(np.abs(y_true_mase - naive_forecast))
    if naive_error == 0:
        naive_error = epsilon
    mase = np.mean(np.abs(y_true_mase - y_pred_mase)) / naive_error
    
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MSE": mse, "MAPE": mape, "sMAPE": smape, "MASE": mase, "R2": r2}

# Funkcja do trenowania i oceny modelu
def train_and_evaluate(df, features, period_name, train_data, test_data, shift_value, model_type="linear"):
    # Przygotowanie danych treningowych i testowych
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    # Przeskalowanie zmiennych objaśniających
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Wybór modelu
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "ridge":
        model = Ridge(alpha=1000.0)  # Możemy później dostroić alpha

    # Trenowanie modelu
    model.fit(X_train, y_train)

    # Prognozowanie
    y_pred = model.predict(X_test)

    # Obliczanie metryk
    metrics = calculate_metrics(y_test, y_pred, shift_value)

    return metrics, y_test, y_pred, test_data["timestamp"]

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

# Trenowanie i ocena dla regresji liniowej
# Pełny zestaw danych
metrics_spokojny_full_linear, y_test_spokojny_full, y_pred_spokojny_full, dates_spokojny_full = train_and_evaluate(
    df_full, features_full, "Spokojny (2019, pełny)", train_spokojny_full, test_spokojny_full, shift_value, model_type="linear"
)
metrics_niespokojny_full_linear, y_test_niespokojny_full, y_pred_niespokojny_full, dates_niespokojny_full = train_and_evaluate(
    df_full, features_full, "Niespokojny (2023, pełny)", train_niespokojny_full, test_niespokojny_full, shift_value, model_type="linear"
)

# Skrócony zestaw danych
metrics_spokojny_short_linear, y_test_spokojny_short, y_pred_spokojny_short, dates_spokojny_short = train_and_evaluate(
    df_short, features_short, "Spokojny (2019, skrócony)", train_spokojny_short, test_spokojny_short, shift_value, model_type="linear"
)
metrics_niespokojny_short_linear, y_test_niespokojny_short, y_pred_niespokojny_short, dates_niespokojny_short = train_and_evaluate(
    df_short, features_short, "Niespokojny (2023, skrócony)", train_niespokojny_short, test_niespokojny_short, shift_value, model_type="linear"
)

# Trenowanie i ocena dla Ridge
# Pełny zestaw danych
metrics_spokojny_full_ridge, _, _, _ = train_and_evaluate(
    df_full, features_full, "Spokojny (2019, pełny)", train_spokojny_full, test_spokojny_full, shift_value, model_type="ridge"
)
metrics_niespokojny_full_ridge, _, _, _ = train_and_evaluate(
    df_full, features_full, "Niespokojny (2023, pełny)", train_niespokojny_full, test_niespokojny_full, shift_value, model_type="ridge"
)

# Skrócony zestaw danych
metrics_spokojny_short_ridge, _, _, _ = train_and_evaluate(
    df_short, features_short, "Spokojny (2019, skrócony)", train_spokojny_short, test_spokojny_short, shift_value, model_type="ridge"
)
metrics_niespokojny_short_ridge, _, _, _ = train_and_evaluate(
    df_short, features_short, "Niespokojny (2023, skrócony)", train_niespokojny_short, test_niespokojny_short, shift_value, model_type="ridge"
)

# Tworzenie DataFrame z wynikami dla regresji liniowej
results_df_linear = pd.DataFrame({
    "Metryka": ["MAE (PLN/MWh)", "RMSE (PLN/MWh)", "MSE (PLN/MWh)^2", "MAPE (%)", "sMAPE (%)", "MASE", "R2"],
    "Spokojny (2019, pełny)": [metrics_spokojny_full_linear["MAE"], metrics_spokojny_full_linear["RMSE"], metrics_spokojny_full_linear["MSE"], 
                               metrics_spokojny_full_linear["MAPE"], metrics_spokojny_full_linear["sMAPE"], metrics_spokojny_full_linear["MASE"], 
                               metrics_spokojny_full_linear["R2"]],
    "Spokojny (2019, skrócony)": [metrics_spokojny_short_linear["MAE"], metrics_spokojny_short_linear["RMSE"], metrics_spokojny_short_linear["MSE"], 
                                  metrics_spokojny_short_linear["MAPE"], metrics_spokojny_short_linear["sMAPE"], metrics_spokojny_short_linear["MASE"], 
                                  metrics_spokojny_short_linear["R2"]],
    "Niespokojny (2023, pełny)": [metrics_niespokojny_full_linear["MAE"], metrics_niespokojny_full_linear["RMSE"], metrics_niespokojny_full_linear["MSE"], 
                                  metrics_niespokojny_full_linear["MAPE"], metrics_niespokojny_full_linear["sMAPE"], metrics_niespokojny_full_linear["MASE"], 
                                  metrics_niespokojny_full_linear["R2"]],
    "Niespokojny (2023, skrócony)": [metrics_niespokojny_short_linear["MAE"], metrics_niespokojny_short_linear["RMSE"], metrics_niespokojny_short_linear["MSE"], 
                                     metrics_niespokojny_short_linear["MAPE"], metrics_niespokojny_short_linear["sMAPE"], metrics_niespokojny_short_linear["MASE"], 
                                     metrics_niespokojny_short_linear["R2"]]
})

# Tworzenie DataFrame z wynikami dla Ridge
results_df_ridge = pd.DataFrame({
    "Metryka": ["MAE (PLN/MWh)", "RMSE (PLN/MWh)", "MSE (PLN/MWh)^2", "MAPE (%)", "sMAPE (%)", "MASE", "R2"],
    "Spokojny (2019, pełny)": [metrics_spokojny_full_ridge["MAE"], metrics_spokojny_full_ridge["RMSE"], metrics_spokojny_full_ridge["MSE"], 
                               metrics_spokojny_full_ridge["MAPE"], metrics_spokojny_full_ridge["sMAPE"], metrics_spokojny_full_ridge["MASE"], 
                               metrics_spokojny_full_ridge["R2"]],
    "Spokojny (2019, skrócony)": [metrics_spokojny_short_ridge["MAE"], metrics_spokojny_short_ridge["RMSE"], metrics_spokojny_short_ridge["MSE"], 
                                  metrics_spokojny_short_ridge["MAPE"], metrics_spokojny_short_ridge["sMAPE"], metrics_spokojny_short_ridge["MASE"], 
                                  metrics_spokojny_short_ridge["R2"]],
    "Niespokojny (2023, pełny)": [metrics_niespokojny_full_ridge["MAE"], metrics_niespokojny_full_ridge["RMSE"], metrics_niespokojny_full_ridge["MSE"], 
                                  metrics_niespokojny_full_ridge["MAPE"], metrics_niespokojny_full_ridge["sMAPE"], metrics_niespokojny_full_ridge["MASE"], 
                                  metrics_niespokojny_full_ridge["R2"]],
    "Niespokojny (2023, skrócony)": [metrics_niespokojny_short_ridge["MAE"], metrics_niespokojny_short_ridge["RMSE"], metrics_niespokojny_short_ridge["MSE"], 
                                     metrics_niespokojny_short_ridge["MAPE"], metrics_niespokojny_short_ridge["sMAPE"], metrics_niespokojny_short_ridge["MASE"], 
                                     metrics_niespokojny_short_ridge["R2"]]
})

# Zaokrąglenie wyników
results_df_linear.iloc[:, 1:] = results_df_linear.iloc[:, 1:].round(2)
results_df_ridge.iloc[:, 1:] = results_df_ridge.iloc[:, 1:].round(2)

# Wyświetlenie wyników
print("Wyniki dla regresji liniowej (po transformacji logarytmicznej):")
print(results_df_linear)
print("\nWyniki dla Ridge (po transformacji logarytmicznej):")
print(results_df_ridge)

# Zapisywanie wyników w formacie LaTeX
latex_table_linear = results_df_linear.to_latex(index=False, float_format="%.2f", 
                                                caption="Wyniki regresji liniowej (po transformacji logarytmicznej) dla pełnego i skróconego zestawu danych.",
                                                label="tab:wyniki_linear_log")
print("\nTabela w formacie LaTeX dla regresji liniowej:")
print(latex_table_linear)

latex_table_ridge = results_df_ridge.to_latex(index=False, float_format="%.2f", 
                                              caption="Wyniki Ridge (po transformacji logarytmicznej) dla pełnego i skróconego zestawu danych.",
                                              label="tab:wyniki_ridge_log")
print("\nTabela w formacie LaTeX dla Ridge:")
print(latex_table_ridge)

# Wykres prognoz vs rzeczywiste wartości dla okresu niespokojnego (2023, regresja liniowa, pełny zestaw)
plt.figure(figsize=(12, 6))
plt.plot(dates_niespokojny_full, np.exp(y_test_niespokojny_full) - shift_value, label='Rzeczywiste ceny', color='blue')
plt.plot(dates_niespokojny_full, np.exp(y_pred_niespokojny_full) - shift_value, label='Prognozy Regresja liniowa (pełny zestaw)', color='red', linestyle='--')
plt.title('Prognozy cen energii (Regresja liniowa po transformacji logarytmicznej, okres niespokojny 2023, pełny zestaw)')
plt.xlabel('Data')
plt.ylabel('Cena (PLN/MWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../../plots/prognozy_linear_log_niespokojny_full.png", dpi=300)
plt.close()

# Wykres prognoz vs rzeczywiste wartości dla okresu niespokojnego (2023, regresja liniowa, skrócony zestaw)
plt.figure(figsize=(12, 6))
plt.plot(dates_niespokojny_short, np.exp(y_test_niespokojny_short) - shift_value, label='Rzeczywiste ceny', color='blue')
plt.plot(dates_niespokojny_short, np.exp(y_pred_niespokojny_short) - shift_value, label='Prognozy Regresja liniowa (skrócony zestaw)', color='red', linestyle='--')
plt.title('Prognozy cen energii (Regresja liniowa po transformacji logarytmicznej, okres niespokojny 2023, skrócony zestaw)')
plt.xlabel('Data')
plt.ylabel('Cena (PLN/MWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../../plots/prognozy_linear_log_niespokojny_short.png", dpi=300)
plt.close()

print("Wykresy zapisane w ../../plots/prognozy_linear_log_niespokojny_full.png i ../../plots/prognozy_linear_log_niespokojny_short.png")
# Importowanie bibliotek
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

# Wczytanie danych
df_full = pd.read_csv("../../data/database.csv")
df_short = pd.read_csv("../../data/short_database.csv")

# Konwersja timestamp na format datetime
df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])
df_short["timestamp"] = pd.to_datetime(df_short["timestamp"])

# Konwersja zmiennych logicznych na numeryczne
df_full["is_holiday"] = df_full["is_holiday"].astype(int)
df_short["is_holiday"] = df_short["is_holiday"].astype(int)

# Lista zmiennych objaśniających dla pełnego zestawu danych (bez gas_volume)
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

# print(f"Przesunięcie danych o: {shift_value}")
# df_full[target] = df_full[target] + shift_value
# df_short[target] = df_short[target] + shift_value

# Zastąpienie ewentualnych wartości ujemnych lub zerowych małą wartością dodatnią
df_full[target] = df_full[target].clip(lower=1e-5)
df_short[target] = df_short[target].clip(lower=1e-5)

# # Transformacja logarytmiczna zmiennej celu
# df_full[target] = np.log(df_full[target])
# df_short[target] = np.log(df_short[target])

# Funkcja do obliczania metryk (bez MASE)
def calculate_metrics(y_true, y_pred, shift_value):
    # Odwrócenie transformacji logarytmicznej i przesunięcia
    # y_true = np.exp(y_true) - shift_value
    # y_pred = np.exp(y_pred) - shift_value

    # Obliczanie MAPE z większym epsilon
    epsilon = 1e-10  # Small value to prevent division by zero
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    
    # Obliczanie sMAPE z zabezpieczeniem przed zerowym mianownikiem
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
    smape = np.mean(np.abs(y_true - y_pred) / denominator) * 100

    # y_true = y_true - shift_value
    # y_pred = y_pred - shift_value

    # Upewniamy się, że y_true i y_pred są pandas.Series
    if isinstance(y_true, np.ndarray):
        y_true = pd.Series(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MSE": mse, "MAPE": mape, "sMAPE": smape, "R2": r2}

# Funkcja do trenowania i oceny modelu MLP
def train_and_evaluate(df, features, period_name, train_data, val_data, test_data, shift_value):
    start_time = time.time()
    
    # Przygotowanie danych treningowych, walidacyjnych i testowych
    X_train = train_data[features]
    y_train = train_data[target]
    X_val = val_data[features]
    y_val = val_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    # Przeskalowanie zmiennych objaśniających
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Inicjalizacja prostego modelu MLP
    model = MLPRegressor(
        hidden_layer_sizes=(32, 32),
        activation='relu',
        solver='adam',
        alpha=0.1,
        max_iter=500,  # Minimalna liczba iteracji, aby przyspieszyć obliczenia
        random_state=42,
        early_stopping=True,  # Włączamy wczesne zatrzymywanie
        validation_fraction=0.01,  # 10% danych treningowych jako wewnętrzny zbiór walidacyjny
        n_iter_no_change=10,  # Zatrzymaj, jeśli brak poprawy przez 10 iteracji
    )

    # Trenowanie modelu
    model.fit(X_train, y_train)

    # Prognozowanie na zbiorze walidacyjnym
    y_pred_val = model.predict(X_val)
    metrics_val = calculate_metrics(y_val, y_pred_val, shift_value)

    # Prognozowanie na zbiorze testowym
    y_pred_test = model.predict(X_test)
    metrics_test = calculate_metrics(y_test, y_pred_test, shift_value)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Czas wykonania dla {period_name}: {execution_time:.2f} sekund")

    return metrics_val, metrics_test, y_test, y_pred_test, test_data["timestamp"]

# Przygotowanie danych dla okresu spokojnego (2016–2019)
# Treningowy: 2016–2018 (75%)
# Walidacyjny: pierwsza połowa 2019
# Testowy: druga połowa 2019
df_spokojny_full = df_full[(df_full["timestamp"].dt.year >= 2016) & (df_full["timestamp"].dt.year <= 2019)]
df_spokojny_short = df_short[(df_short["timestamp"].dt.year >= 2016) & (df_short["timestamp"].dt.year <= 2019)]

# Treningowy: 2016–2018
train_spokojny_full = df_spokojny_full[(df_spokojny_full["timestamp"].dt.year >= 2016) & (df_spokojny_full["timestamp"].dt.year <= 2018)]
train_spokojny_short = df_spokojny_short[(df_spokojny_short["timestamp"].dt.year >= 2016) & (df_spokojny_short["timestamp"].dt.year <= 2018)]

# Podział na treningowy (75%) i wstępny walidacyjny (25%) w ramach 2016–2018
train_spokojny_full = train_spokojny_full.sample(frac=0.75, random_state=42)
train_spokojny_short = train_spokojny_short.sample(frac=0.75, random_state=42)

# Walidacyjny: pierwsza połowa 2019
val_spokojny_full = df_spokojny_full[(df_spokojny_full["timestamp"].dt.year == 2019) & (df_spokojny_full["timestamp"].dt.month <= 6)]
val_spokojny_short = df_spokojny_short[(df_spokojny_short["timestamp"].dt.year == 2019) & (df_spokojny_short["timestamp"].dt.month <= 6)]

# Testowy: druga połowa 2019
test_spokojny_full = df_spokojny_full[(df_spokojny_full["timestamp"].dt.year == 2019) & (df_spokojny_full["timestamp"].dt.month > 6)]
test_spokojny_short = df_spokojny_short[(df_spokojny_short["timestamp"].dt.year == 2019) & (df_spokojny_short["timestamp"].dt.month > 6)]

# Przygotowanie danych dla okresu niespokojnego (2020–2023)
# Treningowy: 2020–2022 (75%)
# Walidacyjny: pierwsza połowa 2023
# Testowy: druga połowa 2023
df_niespokojny_full = df_full[(df_full["timestamp"].dt.year >= 2020) & (df_full["timestamp"].dt.year <= 2023)]
df_niespokojny_short = df_short[(df_short["timestamp"].dt.year >= 2020) & (df_short["timestamp"].dt.year <= 2023)]

# Treningowy: 2020–2022
train_niespokojny_full = df_niespokojny_full[(df_niespokojny_full["timestamp"].dt.year >= 2020) & (df_niespokojny_full["timestamp"].dt.year <= 2022)]
train_niespokojny_short = df_niespokojny_short[(df_niespokojny_short["timestamp"].dt.year >= 2020) & (df_niespokojny_short["timestamp"].dt.year <= 2022)]

# Podział na treningowy (75%) i wstępny walidacyjny (25%) w ramach 2020–2022
train_niespokojny_full = train_niespokojny_full.sample(frac=0.75, random_state=42)
train_niespokojny_short = train_niespokojny_short.sample(frac=0.75, random_state=42)

# Walidacyjny: pierwsza połowa 2023
val_niespokojny_full = df_niespokojny_full[(df_niespokojny_full["timestamp"].dt.year == 2023) & (df_niespokojny_full["timestamp"].dt.month <= 6)]
val_niespokojny_short = df_niespokojny_short[(df_niespokojny_short["timestamp"].dt.year == 2023) & (df_niespokojny_short["timestamp"].dt.month <= 6)]

# Testowy: druga połowa 2023
test_niespokojny_full = df_niespokojny_full[(df_niespokojny_full["timestamp"].dt.year == 2023) & (df_niespokojny_full["timestamp"].dt.month > 6)]
test_niespokojny_short = df_niespokojny_short[(df_niespokojny_short["timestamp"].dt.year == 2023) & (df_niespokojny_short["timestamp"].dt.month > 6)]

# Trenowanie i ocena modelu dla wszystkich zestawów danych
# Spokojny (2019, pełny zestaw)
# metrics_val_spokojny_full, metrics_test_spokojny_full, y_test_spokojny_full, y_pred_spokojny_full, dates_spokojny_full = train_and_evaluate(
#     df_full, features_full, "Spokojny (2019, pełny)", train_spokojny_full, val_spokojny_full, test_spokojny_full, shift_value
# )

# Spokojny (2019, skrócony zestaw)
metrics_val_spokojny_short, metrics_test_spokojny_short, y_test_spokojny_short, y_pred_spokojny_short, dates_spokojny_short = train_and_evaluate(
    df_short, features_short, "Spokojny (2019, skrócony)", train_spokojny_short, val_spokojny_short, test_spokojny_short, shift_value
)

# # Niespokojny (2023, pełny zestaw)
# metrics_val_niespokojny_full, metrics_test_niespokojny_full, y_test_niespokojny_full, y_pred_niespokojny_full, dates_niespokojny_full = train_and_evaluate(
#     df_full, features_full, "Niespokojny (2023, pełny)", train_niespokojny_full, val_niespokojny_full, test_niespokojny_full, shift_value
# )

# Niespokojny (2023, skrócony zestaw)
metrics_val_niespokojny_short, metrics_test_niespokojny_short, y_test_niespokojny_short, y_pred_niespokojny_short, dates_niespokojny_short = train_and_evaluate(
    df_short, features_short, "Niespokojny (2023, skrócony)", train_niespokojny_short, val_niespokojny_short, test_niespokojny_short, shift_value
)

# Tworzenie DataFrame z wynikami dla walidacji (bez MASE)
results_val_df = pd.DataFrame({
    "Metryka": ["MAE (PLN/MWh)", "RMSE (PLN/MWh)", "MSE (PLN/MWh)^2", "MAPE (%)", "sMAPE (%)", "R2"],
    # "Spokojny (2019, pełny)": [metrics_val_spokojny_full["MAE"], metrics_val_spokojny_full["RMSE"], metrics_val_spokojny_full["MSE"], 
    #                            metrics_val_spokojny_full["MAPE"], metrics_val_spokojny_full["sMAPE"], metrics_val_spokojny_full["R2"]],
    "Spokojny (2019, skrócony)": [metrics_val_spokojny_short["MAE"], metrics_val_spokojny_short["RMSE"], metrics_val_spokojny_short["MSE"], 
                                  metrics_val_spokojny_short["MAPE"], metrics_val_spokojny_short["sMAPE"], metrics_val_spokojny_short["R2"]],
    # "Niespokojny (2023, pełny)": [metrics_val_niespokojny_full["MAE"], metrics_val_niespokojny_full["RMSE"], metrics_val_niespokojny_full["MSE"], 
    #                               metrics_val_niespokojny_full["MAPE"], metrics_val_niespokojny_full["sMAPE"], metrics_val_niespokojny_full["R2"]],
    "Niespokojny (2023, skrócony)": [metrics_val_niespokojny_short["MAE"], metrics_val_niespokojny_short["RMSE"], metrics_val_niespokojny_short["MSE"], 
                                     metrics_val_niespokojny_short["MAPE"], metrics_val_niespokojny_short["sMAPE"], metrics_val_niespokojny_short["R2"]]
})

# Tworzenie DataFrame z wynikami dla testu (bez MASE)
results_test_df = pd.DataFrame({
    "Metryka": ["MAE (PLN/MWh)", "RMSE (PLN/MWh)", "MSE (PLN/MWh)^2", "MAPE (%)", "sMAPE (%)", "R2"],
    # "Spokojny (2019, pełny)": [metrics_test_spokojny_full["MAE"], metrics_test_spokojny_full["RMSE"], metrics_test_spokojny_full["MSE"], 
    #                            metrics_test_spokojny_full["MAPE"], metrics_test_spokojny_full["sMAPE"], metrics_test_spokojny_full["R2"]],
    "Spokojny (2019, skrócony)": [metrics_test_spokojny_short["MAE"], metrics_test_spokojny_short["RMSE"], metrics_test_spokojny_short["MSE"], 
                                  metrics_test_spokojny_short["MAPE"], metrics_test_spokojny_short["sMAPE"], metrics_test_spokojny_short["R2"]],
    # "Niespokojny (2023, pełny)": [metrics_test_niespokojny_full["MAE"], metrics_test_niespokojny_full["RMSE"], metrics_test_niespokojny_full["MSE"], 
    #                               metrics_test_niespokojny_full["MAPE"], metrics_test_niespokojny_full["sMAPE"], metrics_test_niespokojny_full["R2"]],
    "Niespokojny (2023, skrócony)": [metrics_test_niespokojny_short["MAE"], metrics_test_niespokojny_short["RMSE"], metrics_test_niespokojny_short["MSE"], 
                                     metrics_test_niespokojny_short["MAPE"], metrics_test_niespokojny_short["sMAPE"], metrics_test_niespokojny_short["R2"]]
})

# Zaokrąglenie wyników
results_val_df.iloc[:, 1:] = results_val_df.iloc[:, 1:].round(2)
results_test_df.iloc[:, 1:] = results_test_df.iloc[:, 1:].round(2)

# Wyświetlenie wyników
print("\nWyniki na zbiorze walidacyjnym:")
print(results_val_df)
print("\nWyniki na zbiorze testowym:")
print(results_test_df)

# # Zapisywanie wyników w formacie LaTeX
# latex_table_val = results_val_df.to_latex(index=False, float_format="%.2f", 
#                                           caption="Wyniki MLP na zbiorze walidacyjnym dla pełnego i skróconego zestawu danych (bez MASE).",
#                                           label="tab:mlp_val_results")
# print("\nTabela w formacie LaTeX dla zbioru walidacyjnego:")
# print(latex_table_val)

# latex_table_test = results_test_df.to_latex(index=False, float_format="%.2f", 
#                                             caption="Wyniki MLP na zbiorze testowym dla pełnego i skróconego zestawu danych (bez MASE).",
#                                             label="tab:mlp_test_results")
# print("\nTabela w formacie LaTeX dla zbioru testowego:")
# print(latex_table_test)

# # Wykres prognoz vs rzeczywiste wartości dla okresu niespokojnego (2023, pełny zestaw, testowy)
# plt.figure(figsize=(12, 6))
# plt.plot(dates_niespokojny_full, np.exp(y_test_niespokojny_full) - shift_value, label='Rzeczywiste ceny', color='blue')
# plt.plot(dates_niespokojny_full, np.exp(y_pred_niespokojny_full) - shift_value, label='Prognozy MLP (pełny zestaw)', color='red', linestyle='--')
# plt.title('Prognozy cen energii (MLP, okres niespokojny 2023, pełny zestaw, zbiór testowy)')
# plt.xlabel('Data')
# plt.ylabel('Cena (PLN/MWh)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("../../plots/prognozy_mlp_niespokojny_full_test.png", dpi=300)
# plt.close()

# # Wykres prognoz vs rzeczywiste wartości dla okresu niespokojnego (2023, skrócony zestaw, testowy)
# plt.figure(figsize=(12, 6))
# plt.plot(dates_niespokojny_short, np.exp(y_test_niespokojny_short) - shift_value, label='Rzeczywiste ceny', color='blue')
# plt.plot(dates_niespokojny_short, np.exp(y_pred_niespokojny_short) - shift_value, label='Prognozy MLP (skrócony zestaw)', color='red', linestyle='--')
# plt.title('Prognozy cen energii (MLP, okres niespokojny 2023, skrócony zestaw, zbiór testowy)')
# plt.xlabel('Data')
# plt.ylabel('Cena (PLN/MWh)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("../../plots/prognozy_mlp_niespokojny_short_test.png", dpi=300)
# plt.close()

# print("Wykresy zapisane w ../../plots/prognozy_mlp_niespokojny_full_test.png i ../../plots/prognozy_mlp_niespokojny_short_test.png")
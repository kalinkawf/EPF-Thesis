import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
from prophet import Prophet

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

# # Transformacja logarytmiczna zmiennej celu
# df_full[target] = np.log(df_full[target])
# df_short[target] = np.log(df_short[target])

# Funkcja do obliczania metryk
def calculate_metrics(y_true, y_pred):
    # y_true = np.exp(y_true) - shift_value
    # y_pred = np.exp(y_pred) - shift_value

    if isinstance(y_true, np.ndarray):
        y_true = pd.Series(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    
    # Przesunięcie wartości, aby uniknąć problemów z MAPE
    min_value = min(y_true.min(), y_pred.min())
    shift = abs(min_value) + 1 if min_value <= 0 else 0
    y_true_shifted = y_true + shift
    y_pred_shifted = y_pred + shift
    
    mape = np.mean(np.abs((y_true_shifted - y_pred_shifted) / y_true_shifted)) * 100
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(np.abs(y_true - y_pred) / np.maximum(denominator, 1e-5)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MSE": mse, "MAPE": mape, "sMAPE": smape, "R2": r2}

# Funkcja do trenowania i oceny modelu Prophet
def train_and_evaluate_prophet(df, features, period_name, train_data, val_data, test_data):
    start_time = time.time()
    
    # Przygotowanie danych w formacie Propheta (ds, y + regresory)
    train_data_prophet = train_data.rename(columns={"timestamp": "ds", target: "y"})[["ds", "y"] + features]
    val_data_prophet = val_data.rename(columns={"timestamp": "ds", target: "y"})[["ds", "y"] + features]
    test_data_prophet = test_data.rename(columns={"timestamp": "ds", target: "y"})[["ds", "y"] + features]

    # Inicjalizacja modelu Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='additive',  # 'additive' lub 'multiplicative' – w Twoim przypadku additive lepiej pasuje do danych
        changepoint_prior_scale=0.01,  # Domyślna wartość, można dostroić
        
    )

    # Dodanie zmiennych objaśniających (regresorów)
    for feature in features:
        model.add_regressor(feature)

    # Trenowanie modelu
    model.fit(train_data_prophet)

    # Prognozowanie na zbiorze walidacyjnym
    future_val = val_data_prophet[["ds"] + features]
    forecast_val = model.predict(future_val)
    y_pred_val = forecast_val["yhat"].values
    y_val = val_data_prophet["y"].values
    metrics_val = calculate_metrics(y_val, y_pred_val)

    # Prognozowanie na zbiorze testowym
    future_test = test_data_prophet[["ds"] + features]
    forecast_test = model.predict(future_test)
    y_pred_test = forecast_test["yhat"].values
    y_test = test_data_prophet["y"].values
    metrics_test = calculate_metrics(y_test, y_pred_test)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Czas wykonania dla {period_name}: {execution_time:.2f} sekund")

    return metrics_val, metrics_test, y_test, y_pred_test, test_data["timestamp"]

# Przygotowanie danych dla okresu spokojnego (2016–2019)
df_spokojny_full = df_full[(df_full["timestamp"].dt.year >= 2016) & (df_full["timestamp"].dt.year <= 2019)]
df_spokojny_short = df_short[(df_short["timestamp"].dt.year >= 2016) & (df_short["timestamp"].dt.year <= 2019)]

train_spokojny_full = df_spokojny_full[(df_spokojny_full["timestamp"].dt.year >= 2016) & (df_spokojny_full["timestamp"].dt.year <= 2018)]
train_spokojny_short = df_spokojny_short[(df_spokojny_short["timestamp"].dt.year >= 2016) & (df_spokojny_short["timestamp"].dt.year <= 2018)]

train_spokojny_full = train_spokojny_full.sample(frac=0.75, random_state=42)
train_spokojny_short = train_spokojny_short.sample(frac=0.75, random_state=42)

val_spokojny_full = df_spokojny_full[(df_spokojny_full["timestamp"].dt.year == 2019) & (df_spokojny_full["timestamp"].dt.month <= 6)]
val_spokojny_short = df_spokojny_short[(df_spokojny_short["timestamp"].dt.year == 2019) & (df_spokojny_short["timestamp"].dt.month <= 6)]

test_spokojny_full = df_spokojny_full[(df_spokojny_full["timestamp"].dt.year == 2019) & (df_spokojny_full["timestamp"].dt.month > 6)]
test_spokojny_short = df_spokojny_short[(df_spokojny_short["timestamp"].dt.year == 2019) & (df_spokojny_short["timestamp"].dt.month > 6)]

# Przygotowanie danych dla okresu niespokojnego (2020–2023)
df_niespokojny_full = df_full[(df_full["timestamp"].dt.year >= 2020) & (df_full["timestamp"].dt.year <= 2023)]
df_niespokojny_short = df_short[(df_short["timestamp"].dt.year >= 2020) & (df_short["timestamp"].dt.year <= 2023)]

train_niespokojny_full = df_niespokojny_full[(df_niespokojny_full["timestamp"].dt.year >= 2020) & (df_niespokojny_full["timestamp"].dt.year <= 2022)]
train_niespokojny_short = df_niespokojny_short[(df_niespokojny_short["timestamp"].dt.year >= 2020) & (df_niespokojny_short["timestamp"].dt.year <= 2022)]

train_niespokojny_full = train_niespokojny_full.sample(frac=0.75, random_state=42)
train_niespokojny_short = train_niespokojny_short.sample(frac=0.75, random_state=42)

val_niespokojny_full = df_niespokojny_full[(df_niespokojny_full["timestamp"].dt.year == 2023) & (df_niespokojny_full["timestamp"].dt.month <= 6)]
val_niespokojny_short = df_niespokojny_short[(df_niespokojny_short["timestamp"].dt.year == 2023) & (df_niespokojny_short["timestamp"].dt.month <= 6)]

test_niespokojny_full = df_niespokojny_full[(df_niespokojny_full["timestamp"].dt.year == 2023) & (df_niespokojny_full["timestamp"].dt.month > 6)]
test_niespokojny_short = df_niespokojny_short[(df_niespokojny_short["timestamp"].dt.year == 2023) & (df_niespokojny_short["timestamp"].dt.month > 6)]

# Listy do przechowywania wyników
results_val = []
results_test = []
period_names = [
    "Spokojny (2019, pełny)",
    "Spokojny (2019, skrócony)",
    "Niespokojny (2023, pełny)",
    "Niespokojny (2023, skrócony)"
]
datasets = [
    (df_full, features_full, train_spokojny_full, val_spokojny_full, test_spokojny_full),
    (df_short, features_short, train_spokojny_short, val_spokojny_short, test_spokojny_short),
    (df_full, features_full, train_niespokojny_full, val_niespokojny_full, test_niespokojny_full),
    (df_short, features_short, train_niespokojny_short, val_niespokojny_short, test_niespokojny_short)
]

# Wyniki dla każdej architektury
val_metrics_all = []
test_metrics_all = []
y_tests = []
y_preds = []
dates_all = []

# Pętla po zestawach danych (4 przypadki)
for (df, features, train_data, val_data, test_data), period_name in zip(datasets, period_names):
    metrics_val, metrics_test, y_test, y_pred_test, dates = train_and_evaluate_prophet(
        df, features, period_name, train_data, val_data, test_data
    )
    val_metrics_all.append(metrics_val)
    test_metrics_all.append(metrics_test)
    y_tests.append(y_test)
    y_preds.append(y_pred_test)
    dates_all.append(dates)

# Tworzenie DataFrame z wynikami dla walidacji
results_val_df = pd.DataFrame({
    "Metryka": ["MAE (PLN/MWh)", "RMSE (PLN/MWh)", "MSE (PLN/MWh)^2", "MAPE (%)", "sMAPE (%)", "R2"],
    "Spokojny (2019, pełny)": [val_metrics_all[0]["MAE"], val_metrics_all[0]["RMSE"], val_metrics_all[0]["MSE"], 
                               val_metrics_all[0]["MAPE"], val_metrics_all[0]["sMAPE"], val_metrics_all[0]["R2"]],
    "Spokojny (2019, skrócony)": [val_metrics_all[1]["MAE"], val_metrics_all[1]["RMSE"], val_metrics_all[1]["MSE"], 
                                  val_metrics_all[1]["MAPE"], val_metrics_all[1]["sMAPE"], val_metrics_all[1]["R2"]],
    "Niespokojny (2023, pełny)": [val_metrics_all[2]["MAE"], val_metrics_all[2]["RMSE"], val_metrics_all[2]["MSE"], 
                                  val_metrics_all[2]["MAPE"], val_metrics_all[2]["sMAPE"], val_metrics_all[2]["R2"]],
    "Niespokojny (2023, skrócony)": [val_metrics_all[3]["MAE"], val_metrics_all[3]["RMSE"], val_metrics_all[3]["MSE"], 
                                     val_metrics_all[3]["MAPE"], val_metrics_all[3]["sMAPE"], val_metrics_all[3]["R2"]]
})

# Tworzenie DataFrame z wynikami dla testu
results_test_df = pd.DataFrame({
    "Metryka": ["MAE (PLN/MWh)", "RMSE (PLN/MWh)", "MSE (PLN/MWh)^2", "MAPE (%)", "sMAPE (%)", "R2"],
    "Spokojny (2019, pełny)": [test_metrics_all[0]["MAE"], test_metrics_all[0]["RMSE"], test_metrics_all[0]["MSE"], 
                               test_metrics_all[0]["MAPE"], test_metrics_all[0]["sMAPE"], test_metrics_all[0]["R2"]],
    "Spokojny (2019, skrócony)": [test_metrics_all[1]["MAE"], test_metrics_all[1]["RMSE"], test_metrics_all[1]["MSE"], 
                                  test_metrics_all[1]["MAPE"], test_metrics_all[1]["sMAPE"], test_metrics_all[1]["R2"]],
    "Niespokojny (2023, pełny)": [test_metrics_all[2]["MAE"], test_metrics_all[2]["RMSE"], test_metrics_all[2]["MSE"], 
                                  test_metrics_all[2]["MAPE"], test_metrics_all[2]["sMAPE"], test_metrics_all[2]["R2"]],
    "Niespokojny (2023, skrócony)": [test_metrics_all[3]["MAE"], test_metrics_all[3]["RMSE"], test_metrics_all[3]["MSE"], 
                                     test_metrics_all[3]["MAPE"], test_metrics_all[3]["sMAPE"], test_metrics_all[3]["R2"]]
})

# Zaokrąglenie wyników
results_val_df.iloc[:, 1:] = results_val_df.iloc[:, 1:].round(2)
results_test_df.iloc[:, 1:] = results_test_df.iloc[:, 1:].round(2)

# Wyświetlenie wyników
print(f"\nWyniki na zbiorze walidacyjnym dla Propheta:")
print(results_val_df)
print(f"\nWyniki na zbiorze testowym dla Propheta:")
print(results_test_df)

# # Wykres prognoz vs rzeczywiste wartości dla okresu niespokojnego (2023, pełny zestaw, testowy)
# plt.figure(figsize=(12, 6))
# plt.plot(dates_all[2], y_tests[2], label='Rzeczywiste ceny', color='blue')
# plt.plot(dates_all[2], y_preds[2], label='Prognozy Prophet (pełny zestaw)', color='red', linestyle='--')
# plt.title('Prognozy cen energii (Prophet, okres niespokojny 2023, pełny zestaw, zbiór testowy)')
# plt.xlabel('Data')
# plt.ylabel('Cena (PLN/MWh)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("../../plots/prognozy_prophet_niespokojny_full_test.png", dpi=300)
# plt.close()

# # Wykres prognoz vs rzeczywiste wartości dla okresu niespokojnego (2023, skrócony zestaw, testowy)
# plt.figure(figsize=(12, 6))
# plt.plot(dates_all[3], y_tests[3], label='Rzeczywiste ceny', color='blue')
# plt.plot(dates_all[3], y_preds[3], label='Prognozy Prophet (skrócony zestaw)', color='red', linestyle='--')
# plt.title('Prognozy cen energii (Prophet, okres niespokojny 2023, skrócony zestaw, zbiór testowy)')
# plt.xlabel('Data')
# plt.ylabel('Cena (PLN/MWh)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("../../plots/prognozy_prophet_niespokojny_short_test.png", dpi=300)
# plt.close()

# # Zapisywanie wyników w formacie LaTeX
# latex_table_test = results_test_df.to_latex(index=False, float_format="%.2f", 
#                                             caption="Wyniki Propheta na zbiorze testowym.",
#                                             label="tab:prophet_test_results")
# print("\nTabela w formacie LaTeX dla zbioru testowego:")
# print(latex_table_test)

# print(f"Wykresy zapisane w ../../plots/ dla Propheta.")
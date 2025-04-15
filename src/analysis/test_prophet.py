# Importowanie bibliotek
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import holidays

# TEST PROPHET LOG 
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
    "pln_usd", "brent_price", "day_of_week", "month", "hour", "fixing_i_price_lag24", "fixing_i_price_lag168"
]

# Lista zmiennych objaśniających dla skróconego zestawu danych
features_short = [
    "fixing_i_price_lag24", "fixing_i_price_lag168",
    "gas_price", "co2_price", "brent_price", "pln_usd", "coal_pscmi1_pln_per_gj",
    "power_loss", "fixing_i_volume", "solar", "gas", "oil", "Load",
    "avg_temp", "avg_wind_speed", "avg_solar_radiation",
    "hour", "month", "wind_onshore", "day_of_week"
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

# Przygotowanie danych w formacie Prophet (ds i y)
df_full_prophet = df_full.rename(columns={"timestamp": "ds", "fixing_i_price": "y"})
df_short_prophet = df_short.rename(columns={"timestamp": "ds", "fixing_i_price": "y"})

def calculate_metrics(y_true, y_pred, shift_value):
    # Odwrócenie transformacji logarytmicznej i przesunięcia
    y_true = np.exp(y_true) - shift_value
    y_pred = np.exp(y_pred) - shift_value

    # y_true = y_true - shift_value
    # y_pred = y_pred - shift_value

    # Upewniamy się, że y_true i y_pred są pandas.Series
    if isinstance(y_true, np.ndarray):
        y_true = pd.Series(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)

    # Podstawowe metryki
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Filtracja wartości nieprawidłowych
    valid_indices = (y_true > 0) & (y_pred > 0)
    y_true = y_true[valid_indices]
    y_pred = y_pred[valid_indices]

    # MAPE z lepszym zabezpieczeniem
    epsilon = 1.0  # Zwiększony epsilon dla lepszego zabezpieczenia
    denominator_mape = np.maximum(np.abs(y_true), epsilon)
    mape = np.mean(np.abs((y_true - y_pred) / denominator_mape)) * 100
    
    # sMAPE z lepszym zabezpieczeniem
    denominator_smape = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2, epsilon)
    smape = np.mean(np.abs(y_true - y_pred) / denominator_smape) * 100

    # MASE z lepszym zabezpieczeniem
    naive_forecast = y_true.shift(1).dropna()
    y_true_mase = y_true[1:]
    y_pred_mase = y_pred[1:]
    
    naive_errors = np.abs(y_true_mase.values - naive_forecast.values)
    naive_error = np.mean(naive_errors)
    
    # Jeśli naive_error jest zbyt mały, ustaw minimalną wartość
    naive_error = max(naive_error, epsilon)
    
    # Obliczenie MASE
    mase = np.mean(np.abs(y_true_mase - y_pred_mase)) / naive_error
    
    return {"MAE": mae, "RMSE": rmse, "MSE": mse, "MAPE": mape, "sMAPE": smape, "MASE": mase, "R2": r2}

# Funkcja do trenowania i oceny modelu Prophet z zadanymi parametrami
def train_and_evaluate(df, features, period_name, train_data, test_data, shift_value, cps, sps, hps):
    # Przygotowanie danych treningowych i testowych
    train_prophet = train_data.rename(columns={"timestamp": "ds", "fixing_i_price": "y"})
    test_prophet = test_data.rename(columns={"timestamp": "ds", "fixing_i_price": "y"})

    # Przeskalowanie regresorów
    scaler = StandardScaler()
    train_prophet[features] = scaler.fit_transform(train_prophet[features])
    test_prophet[features] = scaler.transform(test_prophet[features])

    # Inicjalizacja modelu Prophet z zadanymi parametrami
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=cps,
        seasonality_prior_scale=sps,
        holidays_prior_scale=hps
    )

    # Dodanie niestandardowej sezonowości godzinowej
    model.add_seasonality(name='hourly', period=1/24, fourier_order=10)

    # Dodanie polskich świąt
    pl_holidays = holidays.Poland(years=range(2016, 2024))
    holidays_df = pd.DataFrame({
        'holiday': 'pl_holidays',
        'ds': pd.to_datetime([date for date in pl_holidays.keys()]),
        'lower_window': 0,
        'upper_window': 1
    })
    model.holidays = holidays_df

    # Dodanie zmiennych objaśniających (regresorów)
    for feature in features:
        model.add_regressor(feature)

    # Trenowanie modelu
    model.fit(train_prophet)

    # Prognozowanie na zbiorze testowym
    future = test_prophet.copy()
    forecast = model.predict(future)

    # Obliczanie metryk
    y_true = test_prophet['y']
    y_pred = forecast['yhat']
    metrics = calculate_metrics(y_true, y_pred, shift_value)

    return metrics, y_true, y_pred, test_prophet['ds']

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

# Lista kombinacji parametrów do przetestowania (możesz ją modyfikować)
param_combinations = [
    # {"cps": 0.01, "sps": 1.0, "hps": 10.0},
    # {"cps": 0.1, "sps": 1.0, "hps": 10.0},
    # {"cps": 0.5, "sps": 1.0, "hps": 10.0},
    # {"cps": 0.1, "sps": 5.0, "hps": 10.0},
    # {"cps": 0.1, "sps": 20.0, "hps": 10.0},
    # {"cps": 0.1, "sps": 5.0, "hps": 0.1},
    # {"cps": 0.1, "sps": 5.0, "hps": 20.0},

    # {"cps": 0.1, "sps": 20.0, "hps": 10.0},
    # {"cps": 0.1, "sps": 50.0, "hps": 10.0},
    # {"cps": 0.01, "sps": 20.0, "hps": 10.0},
    # {"cps": 0.01, "sps": 50.0, "hps": 0.01},

    {"cps": 0.01, "sps": 50.0, "hps": 0.1},
    {"cps": 0.1, "sps": 50.0, "hps": 0.1},
    {"cps": 0.1, "sps": 20.0, "hps": 0.1},
]

# Słowniki do przechowywania wyników
results_spokojny_full = []
results_spokojny_short = []
results_niespokojny_full = []
results_niespokojny_short = []

# Przetestowanie każdej kombinacji parametrów
for i, params in enumerate(param_combinations):
    cps = params["cps"]
    sps = params["sps"]
    hps = params["hps"]
    print(f"\nTestowanie kombinacji {i+1}/{len(param_combinations)}: cps={cps}, sps={sps}, hps={hps}")

    # Spokojny (2019, pełny zestaw)
    metrics_spokojny_full, y_test_spokojny_full, y_pred_spokojny_full, dates_spokojny_full = train_and_evaluate(
        df_full, features_full, "Spokojny (2019, pełny)", train_spokojny_full, test_spokojny_full, shift_value, cps, sps, hps
    )
    results_spokojny_full.append({
        "Kombinacja": f"cps={cps}, sps={sps}, hps={hps}",
        "MAE": metrics_spokojny_full["MAE"],
        "RMSE": metrics_spokojny_full["RMSE"],
        "MSE": metrics_spokojny_full["MSE"],
        "MAPE": metrics_spokojny_full["MAPE"],
        "sMAPE": metrics_spokojny_full["sMAPE"],
        "MASE": metrics_spokojny_full["MASE"],
        "R2": metrics_spokojny_full["R2"]
    })

    # Spokojny (2019, skrócony zestaw)
    metrics_spokojny_short, y_test_spokojny_short, y_pred_spokojny_short, dates_spokojny_short = train_and_evaluate(
        df_short, features_short, "Spokojny (2019, skrócony)", train_spokojny_short, test_spokojny_short, shift_value, cps, sps, hps
    )
    results_spokojny_short.append({
        "Kombinacja": f"cps={cps}, sps={sps}, hps={hps}",
        "MAE": metrics_spokojny_short["MAE"],
        "RMSE": metrics_spokojny_short["RMSE"],
        "MSE": metrics_spokojny_short["MSE"],
        "MAPE": metrics_spokojny_short["MAPE"],
        "sMAPE": metrics_spokojny_short["sMAPE"],
        "MASE": metrics_spokojny_short["MASE"],
        "R2": metrics_spokojny_short["R2"]
    })

    # Niespokojny (2023, pełny zestaw)
    metrics_niespokojny_full, y_test_niespokojny_full, y_pred_niespokojny_full, dates_niespokojny_full = train_and_evaluate(
        df_full, features_full, "Niespokojny (2023, pełny)", train_niespokojny_full, test_niespokojny_full, shift_value, cps, sps, hps
    )
    results_niespokojny_full.append({
        "Kombinacja": f"cps={cps}, sps={sps}, hps={hps}",
        "MAE": metrics_niespokojny_full["MAE"],
        "RMSE": metrics_niespokojny_full["RMSE"],
        "MSE": metrics_niespokojny_full["MSE"],
        "MAPE": metrics_niespokojny_full["MAPE"],
        "sMAPE": metrics_niespokojny_full["sMAPE"],
        "MASE": metrics_niespokojny_full["MASE"],
        "R2": metrics_niespokojny_full["R2"]
    })

    # Niespokojny (2023, skrócony zestaw)
    metrics_niespokojny_short, y_test_niespokojny_short, y_pred_niespokojny_short, dates_niespokojny_short = train_and_evaluate(
        df_short, features_short, "Niespokojny (2023, skrócony)", train_niespokojny_short, test_niespokojny_short, shift_value, cps, sps, hps
    )
    results_niespokojny_short.append({
        "Kombinacja": f"cps={cps}, sps={sps}, hps={hps}",
        "MAE": metrics_niespokojny_short["MAE"],
        "RMSE": metrics_niespokojny_short["RMSE"],
        "MSE": metrics_niespokojny_short["MSE"],
        "MAPE": metrics_niespokojny_short["MAPE"],
        "sMAPE": metrics_niespokojny_short["sMAPE"],
        "MASE": metrics_niespokojny_short["MASE"],
        "R2": metrics_niespokojny_short["R2"]
    })

    # Zapisanie wykresów dla każdej kombinacji (tylko dla okresu niespokojnego)
    # Pełny zestaw
    # plt.figure(figsize=(12, 6))
    # plt.plot(dates_niespokojny_full, np.exp(y_test_niespokojny_full) - shift_value, label='Rzeczywiste ceny', color='blue')
    # plt.plot(dates_niespokojny_full, np.exp(y_pred_niespokojny_full) - shift_value, label='Prognozy Prophet (pełny zestaw)', color='red', linestyle='--')
    # plt.title(f'Prognozy cen energii (Prophet, okres niespokojny 2023, pełny zestaw)\nKombinacja: cps={cps}, sps={sps}, hps={hps}')
    # plt.xlabel('Data')
    # plt.ylabel('Cena (PLN/MWh)')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f"../../plots/prognozy_prophet_niespokojny_full_combination_{i+1}.png", dpi=300)
    # plt.close()

    # # Skrócony zestaw
    # plt.figure(figsize=(12, 6))
    # plt.plot(dates_niespokojny_short, np.exp(y_test_niespokojny_short) - shift_value, label='Rzeczywiste ceny', color='blue')
    # plt.plot(dates_niespokojny_short, np.exp(y_pred_niespokojny_short) - shift_value, label='Prognozy Prophet (skrócony zestaw)', color='red', linestyle='--')
    # plt.title(f'Prognozy cen energii (Prophet, okres niespokojny 2023, skrócony zestaw)\nKombinacja: cps={cps}, sps={sps}, hps={hps}')
    # plt.xlabel('Data')
    # plt.ylabel('Cena (PLN/MWh)')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f"../../plots/prognozy_prophet_niespokojny_short_combination_{i+1}.png", dpi=300)
    # plt.close()

# Tworzenie DataFrame z wynikami
results_spokojny_full_df = pd.DataFrame(results_spokojny_full)
results_spokojny_short_df = pd.DataFrame(results_spokojny_short)
results_niespokojny_full_df = pd.DataFrame(results_niespokojny_full)
results_niespokojny_short_df = pd.DataFrame(results_niespokojny_short)

# Zaokrąglenie wyników
results_spokojny_full_df.iloc[:, 1:] = results_spokojny_full_df.iloc[:, 1:].round(2)
results_spokojny_short_df.iloc[:, 1:] = results_spokojny_short_df.iloc[:, 1:].round(2)
results_niespokojny_full_df.iloc[:, 1:] = results_niespokojny_full_df.iloc[:, 1:].round(2)
results_niespokojny_short_df.iloc[:, 1:] = results_niespokojny_short_df.iloc[:, 1:].round(2)

# Wyświetlenie wyników
print("\nWyniki dla Spokojny (2019, pełny zestaw):")
print(results_spokojny_full_df)
print("\nWyniki dla Spokojny (2019, skrócony zestaw):")
print(results_spokojny_short_df)
print("\nWyniki dla Niespokojny (2023, pełny zestaw):")
print(results_niespokojny_full_df)
print("\nWyniki dla Niespokojny (2023, skrócony zestaw):")
print(results_niespokojny_short_df)

# Zapisywanie wyników w formacie LaTeX
# latex_table_spokojny_full = results_spokojny_full_df.to_latex(index=False, float_format="%.2f", 
#                                                               caption="Wyniki dla Prophet (Spokojny 2019, pełny zestaw) dla różnych kombinacji parametrów.",
#                                                               label="tab:prophet_spokojny_full_params")
# print("\nTabela w formacie LaTeX dla Spokojny (2019, pełny zestaw):")
# print(latex_table_spokojny_full)

# latex_table_spokojny_short = results_spokojny_short_df.to_latex(index=False, float_format="%.2f", 
#                                                                 caption="Wyniki dla Prophet (Spokojny 2019, skrócony zestaw) dla różnych kombinacji parametrów.",
#                                                                 label="tab:prophet_spokojny_short_params")
# print("\nTabela w formacie LaTeX dla Spokojny (2019, skrócony zestaw):")
# print(latex_table_spokojny_short)

# latex_table_niespokojny_full = results_niespokojny_full_df.to_latex(index=False, float_format="%.2f", 
#                                                                     caption="Wyniki dla Prophet (Niespokojny 2023, pełny zestaw) dla różnych kombinacji parametrów.",
#                                                                     label="tab:prophet_niespokojny_full_params")
# print("\nTabela w formacie LaTeX dla Niespokojny (2023, pełny zestaw):")
# print(latex_table_niespokojny_full)

# latex_table_niespokojny_short = results_niespokojny_short_df.to_latex(index=False, float_format="%.2f", 
#                                                                       caption="Wyniki dla Prophet (Niespokojny 2023, skrócony zestaw) dla różnych kombinacji parametrów.",
#                                                                       label="tab:prophet_niespokojny_short_params")
# print("\nTabela w formacie LaTeX dla Niespokojny (2023, skrócony zestaw):")
# print(latex_table_niespokojny_short)

# print("\nWykresy zapisane w ../../plots/prognozy_prophet_niespokojny_full_combination_*.png i ../../plots/prognozy_prophet_niespokojny_short_combination_*.png")
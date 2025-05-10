import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import time

# Funkcja do obliczania sMAPE
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# Wczytanie danych
df = pd.read_csv("..\..\data\database.csv")
df2 = pd.read_csv("..\..\data\short_database.csv")  # Upewnij się, że ścieżki są poprawne

# Upewnienie się, że timestamp jest w formacie datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
df2['timestamp'] = pd.to_datetime(df2['timestamp'])

# Definicja pełnego zbioru danych (wszystkie cechy)
full_features = [
    'timestamp', 'temp_waw', 'wind_speed_waw', 'cloud_cover_waw', 'solar_radiation_waw',
    'temp_ksz', 'wind_speed_ksz', 'cloud_cover_ksz', 'solar_radiation_ksz',
    'temp_krk', 'wind_speed_krk', 'cloud_cover_krk', 'solar_radiation_krk',
    'temp_bab', 'wind_speed_bab', 'cloud_cover_bab', 'solar_radiation_bab',
    'power_loss', 'Network_loss', 'Niemcy Bilans', 'Czechy Bilans', 'Litwa Bilans',
    'Słowacja Bilans', 'Szwecja Bilans', 'Ukraina Bilans', 'hard_coal', 'coal-derived',
    'lignite', 'gas', 'oil', 'biomass', 'wind_onshore', 'solar', 'fixing_i_price',
    'fixing_i_volume', 'Load', 'non_emissive_sources_percentage', 'RB_price',
    'cz_price', 'se_price', 'lt_price', 'sk_price', 'gas_price', 'coal_pscmi1_pln_per_gj',
    'co2_price', 'pln_usd', 'pln_eur', 'brent_price', 'day_of_week', 'month', 'hour',
    'fixing_i_price_mean24', 'fixing_i_price_mean48', 'fixing_i_price_lag24',
    'fixing_i_price_lag48', 'fixing_i_price_lag72', 'fixing_i_price_lag96',
    'fixing_i_price_lag120', 'fixing_i_price_lag144', 'fixing_i_price_lag168',
    'is_holiday', 'peak_hour'
]

# Definicja skróconego zbioru danych
short_features = [
    'timestamp', 'fixing_i_price', 'fixing_i_price_lag24', 'fixing_i_price_lag48',
    'fixing_i_price_lag72', 'fixing_i_price_lag96', 'fixing_i_price_lag120',
    'fixing_i_price_lag144', 'fixing_i_price_lag168', 'fixing_i_price_mean24',
    'fixing_i_price_mean48', 'gas_price', 'co2_price', 'brent_price', 'pln_usd',
    'coal_pscmi1_pln_per_gj', 'power_loss', 'fixing_i_volume', 'solar', 'gas', 'oil',
    'hour', 'month', 'is_holiday',
    'non_emissive_sources_percentage', 'day_of_week', 'RB_price', 'se_price',
    'sk_price', 'cz_price', 'lt_price', 'pln_eur'
]

# Sprawdzenie, czy wszystkie cechy są dostępne w danych
missing_full = [col for col in full_features if col not in df.columns]
missing_short = [col for col in short_features if col not in df2.columns]
if missing_full:
    print(f"Brakujące cechy w pełnym zbiorze: {missing_full}")
if missing_short:
    print(f"Brakujące cechy w skróconym zbiorze: {missing_short}")

# Wybór cech dla pełnego i skróconego zbioru
df_full = df[full_features].copy()
df_short = df2[short_features].copy()

# Sprawdzenie NaN w oryginalnych danych
print("\n### Sprawdzenie NaN w oryginalnym pełnym zbiorze ###")
print(df_full.isna().sum()[df_full.isna().sum() > 0])

# Preprocessing: Kodowanie cykliczne zmiennych hour, day_of_week, month
def encode_cyclic(df, column, max_value):
    df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / max_value)
    df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / max_value)
    return df

# Kodowanie zmiennych cyklicznych
for df in [df_full, df_short]:
    # Hour (0-23)
    df = encode_cyclic(df, 'hour', 24)
    # Day of week (0-6)
    df = encode_cyclic(df, 'day_of_week', 7)
    # Month (1-12)
    df['month'] = df['month'].astype(int)  # Upewniamy się, że month jest liczbą
    df = encode_cyclic(df, 'month', 12)

# Usunięcie oryginalnych zmiennych cyklicznych (zastąpione przez sin/cos)
for df in [df_full, df_short]:
    df.drop(columns=['hour', 'day_of_week', 'month'], inplace=True)

# Definicja zmiennych do standaryzacji (wszystkie numeryczne poza fixing_i_price, timestamp, is_holiday, peak_hour)
features_to_scale_full = [col for col in df_full.columns if col not in ['timestamp', 'fixing_i_price', 'is_holiday', 'peak_hour']]
features_to_scale_short = [col for col in df_short.columns if col not in ['timestamp', 'fixing_i_price', 'is_holiday']]

# Standaryzacja zmiennych
scaler_full = StandardScaler()
scaler_short = StandardScaler()

df_full[features_to_scale_full] = scaler_full.fit_transform(df_full[features_to_scale_full])
df_short[features_to_scale_short] = scaler_short.fit_transform(df_short[features_to_scale_short])

# Sprawdzenie NaN po standaryzacji
print("\n### Sprawdzenie NaN po standaryzacji w pełnym zbiorze ###")
nan_count_full = df_full.isna().sum()
print(nan_count_full[nan_count_full > 0])
if nan_count_full.sum() > 0:
    print(f"Całkowita liczba NaN po standaryzacji: {nan_count_full.sum()}")

print("\n### Sprawdzenie NaN po standaryzacji w skróconym zbiorze ###")
nan_count_short = df_short.isna().sum()
print(nan_count_short[nan_count_short > 0])
if nan_count_short.sum() > 0:
    print(f"Całkowita liczba NaN po standaryzacji: {nan_count_short.sum()}")

# Ustawienie indeksu na timestamp
df_full.set_index('timestamp', inplace=True)
df_short.set_index('timestamp', inplace=True)

# Filtrowanie danych: trenowanie 2016-2018, testowanie 2019
train_full = df_full['2016-01-01':'2018-12-31'].reset_index()
test_full = df_full['2019-01-01':'2019-12-31'].reset_index()

# Diagnostyka danych testowych
print(f"Liczba wierszy w test_full: {len(test_full)}")
print(f"Liczba unikalnych timestampów w test_full: {test_full['timestamp'].nunique()}")
print(f"Zakres timestampów w test_full: {test_full['timestamp'].min()} do {test_full['timestamp'].max()}")

train_short = df_short['2016-01-01':'2018-12-31'].reset_index()
test_short = df_short['2019-01-01':'2019-12-31'].reset_index()

# Przygotowanie danych dla Prophet
train_prophet_full = train_full.rename(columns={'timestamp': 'ds', 'fixing_i_price': 'y'})
test_prophet_full = test_full.rename(columns={'timestamp': 'ds', 'fixing_i_price': 'y'})
train_prophet_short = train_short.rename(columns={'timestamp': 'ds', 'fixing_i_price': 'y'})
test_prophet_short = test_short.rename(columns={'timestamp': 'ds', 'fixing_i_price': 'y'})

# Definicja regresorów (wszystkie cechy poza 'ds', 'y' i zmiennymi czasowymi)
regressors_full = [col for col in train_prophet_full.columns if col not in ['ds', 'y', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'is_holiday', 'peak_hour']]
regressors_short = [col for col in train_prophet_short.columns if col not in ['ds', 'y', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'is_holiday']]

# Sprawdzenie NaN w danych treningowych i testowych
print("\n### Sprawdzenie NaN w danych treningowych i testowych (pełny zbiór) ###")
print("Treningowe NaN:", train_prophet_full[regressors_full].isna().sum().sum())
print("Testowe NaN:", test_prophet_full[regressors_full].isna().sum().sum())

print("\n### Sprawdzenie NaN w danych treningowych i testowych (skrócony zbiór) ###")
print("Treningowe NaN:", train_prophet_short[regressors_short].isna().sum().sum())
print("Testowe NaN:", test_prophet_short[regressors_short].isna().sum().sum())

# Wypełnienie NaN w danych testowych średnią z danych treningowych (na wszelki wypadek)
for col in regressors_full:
    if test_prophet_full[col].isna().sum() > 0:
        test_prophet_full[col] = test_prophet_full[col].fillna(train_prophet_full[col].mean())
for col in regressors_short:
    if test_prophet_short[col].isna().sum() > 0:
        test_prophet_short[col] = test_prophet_short[col].fillna(train_prophet_short[col].mean())

# Definicja polskich świąt
polish_holidays = pd.DataFrame({
    'holiday': 'polish_holidays',
    'ds': pd.to_datetime([
        # Stałe święta w Polsce (2016-2019)
        '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01',  # Nowy Rok
        '2016-01-06', '2017-01-06', '2018-01-06', '2019-01-06',  # Trzech Króli
        '2016-05-01', '2017-05-01', '2018-05-01', '2019-05-01',  # Święto Pracy
        '2016-05-03', '2017-05-03', '2018-05-03', '2019-05-03',  # Konstytucja 3 Maja
        '2016-08-15', '2017-08-15', '2018-08-15', '2019-08-15',  # Wniebowzięcie NMP
        '2016-11-01', '2017-11-01', '2018-11-01', '2019-11-01',  # Wszystkich Świętych
        '2016-11-11', '2017-11-11', '2018-11-11', '2019-11-11',  # Dzień Niepodległości
        '2016-12-25', '2017-12-25', '2018-12-25', '2019-12-25',  # Boże Narodzenie (pierwszy dzień)
        '2016-12-26', '2017-12-26', '2018-12-26', '2019-12-26',  # Boże Narodzenie (drugi dzień)
        # Wielkanoc i Zielone Świątki (zmienne daty w zależności od roku)
        '2016-03-27', '2017-04-16', '2018-04-01', '2019-04-21',  # Wielkanoc (niedziela)
        '2016-03-28', '2017-04-17', '2018-04-02', '2019-04-22',  # Poniedziałek Wielkanocny
        '2016-05-15', '2017-06-04', '2018-05-20', '2019-06-09',  # Zielone Świątki (Zesłanie Ducha Świętego)
        '2016-05-26', '2017-06-15', '2018-05-31', '2019-06-20',  # Boże Ciało
    ]),
    'lower_window': 0,  # Dzień święta
    'upper_window': 1,  # Dzień po święcie
})

# Definicja kombinacji parametrów Prophet
param_combinations = [
    {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 20.0, 'holidays_prior_scale': 0.1, 'seasonality_mode': 'additive'},
    # {'changepoint_prior_scale': 0.0001, 'seasonality_prior_scale': 20.0, 'holidays_prior_scale': 10.0, 'seasonality_mode': 'additive'},
    # {'changepoint_prior_scale': 0.0001, 'seasonality_prior_scale': 50.0, 'holidays_prior_scale': 0.1, 'seasonality_mode': 'additive'},
    # {'changepoint_prior_scale': 0.001, 'seasonality_prior_scale': 0.1, 'holidays_prior_scale': 5.0, 'seasonality_mode': 'additive'},
    # {'changepoint_prior_scale': 0.001, 'seasonality_prior_scale': 1.0, 'holidays_prior_scale': 10.0, 'seasonality_mode': 'additive'},
]

# Funkcja do trenowania i oceny modelu Prophet
def evaluate_prophet(train_data, test_data, params, combination_idx, dataset_type, regressors):
    print(f"Testowanie kombinacji {combination_idx} ({dataset_type}): {params}")

    start_time = time.time()
    
    # Inicjalizacja modelu Prophet z polskimi świętami
    model = Prophet(**params, holidays=polish_holidays, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    
    # Dodanie regresorów
    for regressor in regressors:
        model.add_regressor(regressor)
    
    # Trenowanie modelu
    model.fit(train_data)
    
    # Użycie danych testowych jako future
    future = test_data[['ds'] + regressors].copy()
    
    # Debug: Sprawdzenie NaN w future
    nan_cols = future.columns[future.isna().any()].tolist()
    if nan_cols:
        print(f"Znaleziono NaN w kolumnach future: {nan_cols}")
    
    # Sprawdzenie ciągłości timestampów w future
    if len(future) != len(test_data):
        print(f"Uwaga: Liczba wierszy w future ({len(future)}) różni się od test_data ({len(test_data)})")
    
    forecast = model.predict(future)
    
    # Obliczenie metryk
    y_true = test_data['y'].values
    y_pred = forecast['yhat'].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    smape_value = smape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Zmierz czas zakończenia
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Czas wykonania dla kombinacji {combination_idx} ({dataset_type}): {execution_time:.2f} sekund")
    
    # # Histogram reszt
    # residuals = y_true - y_pred
    # plt.figure(figsize=(6, 4))
    # plt.hist(residuals, bins=50, color="red", alpha=0.7)
    # plt.title(f"Histogram reszt - Prophet", fontsize=12)
    # plt.xlabel("Reszty", fontsize=10)
    # plt.ylabel("Częstość", fontsize=10)
    # plt.grid(True)
    # plt.tight_layout()
    # os.makedirs('../../plots/predicts', exist_ok=True)
    # plt.savefig(f'../../plots/predicts/residuals_histogram_Prophet_stable.png', dpi=300)
    # plt.close()
    
    # # Wykres predykcji vs. wartości rzeczywistych
    # # Filtrowanie danych testowych dla Q1 (pierwszy kwartał)
    # test_data_q1 = test_data[(test_data['ds'] >= '2019-01-01') & (test_data['ds'] < '2019-04-01')]
    # y_true_q1 = test_data_q1['y'].values
    # y_pred_q1 = forecast.loc[forecast['ds'].isin(test_data_q1['ds']), 'yhat'].values

    # plt.figure(figsize=(12, 4))
    # plt.plot(test_data_q1['ds'], y_true_q1, label="Rzeczywiste", color="blue", alpha=0.7)
    # plt.plot(test_data_q1['ds'], y_pred_q1, label="Prophet", color="red", alpha=0.7, linestyle='--')
    # plt.title(f"Okres stabilny - Q1 2019", fontsize=14)
    # plt.xlabel("Czas", fontsize=12)
    # plt.ylabel("Cena energii [PLN/MWh]", fontsize=12)
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f'../../plots/predicts/Prophet_predictions_stable_Q1.png', dpi=300)
    # plt.close()

    # # --- Wykres błędów na całym okresie testowym ---
    # plt.figure(figsize=(12, 6))
    # plt.plot(test_data["ds"], y_true - y_pred, color="red", alpha=0.7, label="Błędy")
    # plt.axhline(y=0, color="black", linestyle="--", label="Linia zerowa")
    # plt.title("Błędy na całym okresie testowym - Prophet", fontsize=14)
    # plt.xlabel("Czas", fontsize=12)
    # plt.ylabel("Błędy w czasie [PLN/MWh]", fontsize=12)
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("../../plots/predicts/errors_over_time_Prophet_full_stable_period.png", dpi=300)
    # plt.close()

    # --- Wykres top 50 błędów (Prophet, pełny zbiór) ---
    # residuals_abs = np.abs(residuals)
    # top_50_indices = np.argsort(residuals_abs)[-50:]  # Indeksy 50 największych błędów
    # top_50_actual = y_true[top_50_indices]
    # top_50_pred = y_pred[top_50_indices]
    # top_50_timestamps = test_data.iloc[top_50_indices]["ds"].values

    # # Wyprintowanie timestampów największych błędów
    # print("\nTimestampy największych błędów:")
    # print(top_50_timestamps)

    # plt.figure(figsize=(8, 6))
    # plt.scatter(top_50_pred, top_50_actual, color="red", alpha=0.7)
    # plt.plot([min(top_50_actual.min(), top_50_pred.min()), max(top_50_actual.max(), top_50_pred.max())],
    #          [min(top_50_actual.min(), top_50_pred.min()), max(top_50_actual.max(), top_50_pred.max())],
    #          color="black", linestyle="--", label="Linia idealna")
    # plt.title("Top 50 błędów: Predykcje vs Rzeczywiste (Prophet, pełny zbiór)", fontsize=12)
    # plt.xlabel("Predykcje [PLN/MWh]", fontsize=10)
    # plt.ylabel("Rzeczywiste wartości [PLN/MWh]", fontsize=10)
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("../../plots/predicts/top_50_errors_Prophet_full_stable_period.png", dpi=300)
    # plt.close()
    
    return {
        'Combination': combination_idx,
        'Changepoint_prior_scale': params['changepoint_prior_scale'],
        'Seasonality_prior_scale': params['seasonality_prior_scale'],
        'Holidays_prior_scale': params['holidays_prior_scale'],
        'Seasonality_mode': params['seasonality_mode'],
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape,
        'sMAPE (%)': smape_value,
        'R2': r2,
    }

# Testowanie na pełnym i skróconym zbiorze danych
results_full = []
results_short = []

for i, params in enumerate(param_combinations, 1):
    # Pełny zbiór danych
    result_full = evaluate_prophet(train_prophet_full, test_prophet_full, params, i, 'Pełny', regressors_full)
    results_full.append(result_full)
    
    # Skrócony zbiór danych
    result_short = evaluate_prophet(train_prophet_short, test_prophet_short, params, i, 'Skrócony', regressors_short)
    results_short.append(result_short)

# Konwersja wyników do DataFrame
results_full_df = pd.DataFrame(results_full)
results_short_df = pd.DataFrame(results_short)

# Zapis wyników do plików CSV
results_full_df.to_csv('../../plots/predicts/prophet_results_full_stable.csv', index=False)
results_short_df.to_csv('../../plots/predicts/prophet_results_short_stable.csv', index=False)

print("\nWyniki dla pełnego zbioru danych:")
print(results_full_df)
print("\nWyniki dla skróconego zbioru danych:")
print(results_short_df)
# mlp_keras_grid_search_unstable.py

# Importowanie bibliotek
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import os

# Funkcja do obliczania sMAPE
def smape(y_true, y_pred):
    epsilon = 1e-5
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_true - y_pred) / np.maximum(denominator, epsilon)) * 100

# Wczytanie danych
df_full = pd.read_csv("../../data/database.csv")
df_short = pd.read_csv("../../data/short_database.csv")

# Konwersja timestamp na format datetime
df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])
df_short["timestamp"] = pd.to_datetime(df_short["timestamp"])

# Konwersja zmiennych logicznych na numeryczne
df_full["is_holiday"] = df_full["is_holiday"].astype(int)
df_full["peak_hour"] = df_full["peak_hour"].astype(int)
df_short["is_holiday"] = df_short["is_holiday"].astype(int)

# Definicja pełnego i skróconego zbioru danych
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

short_features = [
    'timestamp', 'fixing_i_price', 'fixing_i_price_lag24', 'fixing_i_price_lag48',
    'fixing_i_price_lag72', 'fixing_i_price_lag96', 'fixing_i_price_lag120',
    'fixing_i_price_lag144', 'fixing_i_price_lag168', 'fixing_i_price_mean24',
    'fixing_i_price_mean48', 'gas_price', 'co2_price', 'brent_price', 'pln_usd',
    'coal_pscmi1_pln_per_gj', 'power_loss', 'fixing_i_volume', 'solar', 'gas', 'oil',
    'Load', 'hour', 'month', 'is_holiday',
    'non_emissive_sources_percentage', 'day_of_week', 'RB_price', 'se_price',
    'sk_price', 'cz_price', 'lt_price', 'pln_eur'
]

target = "fixing_i_price"

# Usunięcie brakujących wartości
df_full = df_full[full_features].dropna()
df_short = df_short[short_features].dropna()

# Preprocessing: Kodowanie cykliczne
def encode_cyclic(df, column, max_value):
    df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / max_value)
    df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / max_value)
    return df

for df in [df_full, df_short]:
    df = encode_cyclic(df, 'hour', 24)
    df = encode_cyclic(df, 'day_of_week', 7)
    df = encode_cyclic(df, 'month', 12)
    df.drop(columns=['hour', 'day_of_week', 'month'], inplace=True)

# Aktualizacja list cech po kodowaniu cyklicznym
full_features_updated = [col for col in df_full.columns if col not in ['timestamp', target]]
short_features_updated = [col for col in df_short.columns if col not in ['timestamp', target]]

# Definicja zmiennych do standaryzacji
features_to_scale_full = [col for col in df_full.columns if col not in ['timestamp', target, 'is_holiday', 'peak_hour']]
features_to_scale_short = [col for col in df_short.columns if col not in ['timestamp', target, 'is_holiday']]

# Standaryzacja zmiennych
scaler_full = StandardScaler()
scaler_short = StandardScaler()
df_full[features_to_scale_full] = scaler_full.fit_transform(df_full[features_to_scale_full])
df_short[features_to_scale_short] = scaler_short.fit_transform(df_short[features_to_scale_short])

# Przygotowanie danych dla okresu niestabilnego
df_unstable_full = df_full[df_full["timestamp"].dt.year >= 2020]
df_unstable_short = df_short[df_short["timestamp"].dt.year >= 2020]

train_unstable_full = df_unstable_full[(df_unstable_full["timestamp"].dt.year >= 2020) & (df_unstable_full["timestamp"].dt.year <= 2022)]
train_unstable_short = df_unstable_short[(df_unstable_short["timestamp"].dt.year >= 2020) & (df_unstable_short["timestamp"].dt.year <= 2022)]

test_unstable_full = df_unstable_full[df_unstable_full["timestamp"].dt.year == 2023]
test_unstable_short = df_unstable_short[df_unstable_short["timestamp"].dt.year == 2023]

# Lista architektur do przetestowania
architectures = [
    (64, 64, 32, 16, 8),
]

# Funkcja do trenowania i oceny modelu MLP w Keras
def train_and_evaluate_keras(df, features, period_name, train_data, test_data, architecture):
    start_time = time.time()
    features = [feature for feature in features if feature != "timestamp"]

    # Przygotowanie danych
    X_train = train_data[features].values
    y_train = train_data[target].values
    X_test = test_data[features].values
    y_test = test_data[target].values

    # Standaryzacja zmiennych
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Debug: Sprawdzenie wymiarów y_train i y_test
    print(f"{period_name} - y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    # Budowa modelu w Keras
    model = Sequential()
    model.add(Dense(architecture[0], activation='relu', input_dim=X_train.shape[1], kernel_regularizer=l2(0.1)))
    for units in architecture[1:]:
        model.add(Dense(units, activation='relu', kernel_regularizer=l2(0.1)))
        model.add(Dropout(0.2))
    model.add(Dense(1))

    # Kompilacja modelu
    model.compile(optimizer=Adam(learning_rate=0.005), loss='mse')

    # Trenowanie modelu
    model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=64,
        verbose=0
    )

    # Prognozowanie
    y_pred_test = model.predict(X_test, verbose=0).flatten()

    # Debug: Sprawdzenie wymiarów y_test i y_pred_test
    print(f"{period_name} - y_test shape: {y_test.shape}, y_pred_test shape: {y_pred_test.shape}")

    # Obliczenie metryk
    metrics_test = {
        "MAE": mean_absolute_error(y_test, y_pred_test),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred_test) * 100,
        "sMAPE": smape(y_test, y_pred_test),
        "R2": r2_score(y_test, y_pred_test)
    }

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Czas wykonania dla {period_name} (architektura {architecture}): {execution_time:.2f} sekund")

    return metrics_test, y_test, y_pred_test, test_data["timestamp"]

# Lista do przechowywania wyników
results_test = []
period_names = ["Niestabilny (2023, pełny)", "Niestabilny (2023, skrócony)"]
datasets = [
    (df_full, full_features_updated, train_unstable_full, test_unstable_full),
    (df_short, short_features_updated, train_unstable_short, test_unstable_short)
]

# Pętla po architekturach
for architecture in architectures:
    print(f"\nTestowanie architektury: {architecture}")
    
    # Wyniki dla każdej architektury
    test_metrics_all = []
    y_tests = []
    y_preds = []
    dates_all = []

    # Pętla po zestawach danych
    for (df, features, train_data, test_data), period_name in zip(datasets, period_names):
        metrics_test, y_test, y_pred_test, dates = train_and_evaluate_keras(
            df, features, period_name, train_data, test_data, architecture
        )
        test_metrics_all.append(metrics_test)
        y_tests.append(y_test)
        y_preds.append(y_pred_test)
        dates_all.append(dates)

    # Tworzenie DataFrame z wynikami dla testu
    results_test_df = pd.DataFrame({
        "Metryka": ["MAE (PLN/MWh)", "RMSE (PLN/MWh)", "MAPE (%)", "sMAPE (%)", "R2"],
        "Niestabilny (2023, pełny)": [test_metrics_all[0]["MAE"], test_metrics_all[0]["RMSE"], test_metrics_all[0]["MAPE"], 
                                      test_metrics_all[0]["sMAPE"], test_metrics_all[0]["R2"]],
        "Niestabilny (2023, skrócony)": [test_metrics_all[1]["MAE"], test_metrics_all[1]["RMSE"], test_metrics_all[1]["MAPE"], 
                                         test_metrics_all[1]["sMAPE"], test_metrics_all[1]["R2"]]
    })

    # Zaokrąglenie wyników
    results_test_df.iloc[:, 1:] = results_test_df.iloc[:, 1:].round(2)

    # Dodanie wyników do listy
    results_test.append((architecture, results_test_df))

    # Wyświetlenie wyników
    print(f"\nWyniki na zbiorze testowym dla architektury {architecture}:")
    print(results_test_df)

    # Zapisywanie wyników do pliku
    os.makedirs('../../plots/mlp2', exist_ok=True)
    results_test_df.to_csv(f'../../plots/mlp2/mlp_results_{architecture}.csv', index=False)

    # Wykresy predykcji vs rzeczywiste wartości (Aug-Sep 2023 dla niestabilnego)
    unstable_aug_sep_dates = dates_all[0][(dates_all[0].dt.month >= 8) & (dates_all[0].dt.month <= 9)]
    unstable_aug_sep_y_test = y_tests[0][(dates_all[0].dt.month >= 8) & (dates_all[0].dt.month <= 9)]
    unstable_aug_sep_y_pred = y_preds[0][(dates_all[0].dt.month >= 8) & (dates_all[0].dt.month <= 9)]
    plt.figure(figsize=(12, 4))
    plt.plot(unstable_aug_sep_dates, unstable_aug_sep_y_test, label='Rzeczywiste ceny', color='blue', alpha=0.7)
    plt.plot(unstable_aug_sep_dates, unstable_aug_sep_y_pred, label='MLP', color='red', alpha=0.7, linestyle='--')
    plt.title('Okres niestabilny (Sierpień-Wrzesień 2023)', fontsize=14)
    plt.xlabel('Czas', fontsize=12)
    plt.ylabel('Cena energii [PLN/MWh]', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../../plots/mlp2/mlp_predictions_full_unstable_aug_sep_{architecture}.png', dpi=300)
    plt.close()

    # Niestabilny (Aug-Sep 2023, skrócony zestaw)
    unstable_aug_sep_short_dates = dates_all[1][(dates_all[1].dt.month >= 8) & (dates_all[1].dt.month <= 9)]
    unstable_aug_sep_short_y_test = y_tests[1][(dates_all[1].dt.month >= 8) & (dates_all[1].dt.month <= 9)]
    unstable_aug_sep_short_y_pred = y_preds[1][(dates_all[1].dt.month >= 8) & (dates_all[1].dt.month <= 9)]
    plt.figure(figsize=(12, 4))
    plt.plot(unstable_aug_sep_short_dates, unstable_aug_sep_short_y_test, label='Rzeczywiste ceny', color='blue', alpha=0.7)
    plt.plot(unstable_aug_sep_short_dates, unstable_aug_sep_short_y_pred, label='MLP', color='red', alpha=0.7, linestyle='--')
    plt.title('Okres niestabilny (Sierpień-Wrzesień 2023, skrócony zestaw)', fontsize=14)
    plt.xlabel('Czas', fontsize=12)
    plt.ylabel('Cena energii [PLN/MWh]', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../../plots/mlp2/mlp_predictions_short_unstable_aug_sep_{architecture}.png', dpi=300)
    plt.close()

    # Histogram błędów (dla całego zbioru testowego niestabilnego 2023, pełny zestaw)
    errors_unstable = y_tests[0] - y_preds[0]
    plt.figure(figsize=(8, 6))
    plt.hist(errors_unstable, bins=50, color='red', alpha=0.7)
    plt.title('Histogram reszt - MLP', fontsize=14)
    plt.xlabel('Reszty [PLN/MWh]', fontsize=12)
    plt.ylabel('Częstość', fontsize=12)
    plt.grid(True)
    plt.savefig(f'../../plots/mlp2/mlp_errors_histogram_full_unstable_{architecture}.png', dpi=300)
    plt.close()

    # Histogram błędów (dla całego zbioru testowego niestabilnego 2023, skrócony zestaw)
    errors_unstable_short = y_tests[1] - y_preds[1]
    plt.figure(figsize=(8, 6))
    plt.hist(errors_unstable_short, bins=50, color='red', alpha=0.7)
    plt.title('Histogram reszt - MLP', fontsize=14)
    plt.xlabel('Reszty [PLN/MWh]', fontsize=12)
    plt.ylabel('Częstość', fontsize=12)
    plt.grid(True)
    plt.savefig(f'../../plots/mlp2/mlp_errors_histogram_short_unstable_{architecture}.png', dpi=300)
    plt.close()

    # Wykres błędów na całym okresie testowym
    for i, period_name in enumerate(period_names):
        residuals = y_tests[i] - y_preds[i]
        plt.figure(figsize=(12, 6))
        plt.plot(dates_all[i], residuals, color="red", alpha=0.7, label="Błędy")
        plt.axhline(y=0, color="black", linestyle="--", label="Linia zerowa")
        plt.title("Błędy na całym okresie testowym - MLP", fontsize=14)
        plt.xlabel("Czas", fontsize=12)
        plt.ylabel("Błędy w czasie [PLN/MWh]", fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"../../plots/mlp2/errors_over_time_{architecture}_{period_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.png", dpi=300)
        plt.close()
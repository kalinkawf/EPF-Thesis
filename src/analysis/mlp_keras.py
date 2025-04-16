# mlp_keras_grid_search_full.py

# Importowanie bibliotek
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

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

# Funkcja do obliczania metryk
def calculate_metrics(y_true, y_pred):
    if isinstance(y_true, np.ndarray):
        y_true = pd.Series(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    
    epsilon = 1e-5
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(np.abs(y_true - y_pred) / np.maximum(denominator, epsilon)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MSE": mse, "MAPE": mape, "sMAPE": smape, "R2": r2}

# Funkcja do trenowania i oceny modelu MLP w Keras
def train_and_evaluate_keras(df, features, period_name, train_data, val_data, test_data, architecture):
    start_time = time.time()
    
    # Przygotowanie danych
    X_train = train_data[features].values
    y_train = train_data[target].values
    X_val = val_data[features].values
    y_val = val_data[target].values
    X_test = test_data[features].values
    y_test = test_data[target].values

    # Przeskalowanie zmiennych objaśniających
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Budowa modelu w Keras
    model = Sequential()
    # Pierwsza warstwa (z input_dim)
    model.add(Dense(architecture[0], activation='relu', input_dim=X_train.shape[1], kernel_regularizer=l2(0.1)))
    # Dodatkowe warstwy (jeśli istnieją)
    for units in architecture[1:]:
        model.add(Dense(units, activation='relu', kernel_regularizer=l2(0.1)))
        model.add(Dropout(0.2))  # Dodanie warstwy Dropout
    # Warstwa wyjściowa
    model.add(Dense(1))

    # Kompilacja modelu
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

    # Wczesne zatrzymywanie
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True
    )

    # Trenowanie modelu
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=500,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=0
    )

    # Prognozowanie
    y_pred_val = model.predict(X_val, verbose=0).flatten()
    metrics_val = calculate_metrics(y_val, y_pred_val)

    y_pred_test = model.predict(X_test, verbose=0).flatten()
    metrics_test = calculate_metrics(y_test, y_pred_test)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Czas wykonania dla {period_name} (architektura {architecture}): {execution_time:.2f} sekund")

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

# Lista architektur do przetestowania
architectures = [
    # 1 warstwa
    # (32,),
    # (64,),
    # 2 warstwy
    # (32, 32),
    # (64, 32),
    # 3 warstwy
    # (32, 32, 32),
    # (64, 32, 16)
    # (32, 32, 16),
    # (32, 16, 8),
    # (128, 64, 32),
    # (64, 32, 16, 8),
    (32, 32, 16, 8),
    (32, 16, 16, 8),
]

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

# Pętla po architekturach
for architecture in architectures:
    print(f"\nTestowanie architektury: {architecture}")
    
    # Wyniki dla każdej architektury
    val_metrics_all = []
    test_metrics_all = []
    y_tests = []
    y_preds = []
    dates_all = []

    # Pętla po zestawach danych (4 przypadki)
    for (df, features, train_data, val_data, test_data), period_name in zip(datasets, period_names):
        metrics_val, metrics_test, y_test, y_pred_test, dates = train_and_evaluate_keras(
            df, features, period_name, train_data, val_data, test_data, architecture
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

    # Dodanie wyników do listy
    results_val.append((architecture, results_val_df))
    results_test.append((architecture, results_test_df))

    # Wyświetlenie wyników dla bieżącej architektury
    print(f"\nWyniki na zbiorze walidacyjnym dla architektury {architecture}:")
    print(results_val_df)
    print(f"\nWyniki na zbiorze testowym dla architektury {architecture}:")
    print(results_test_df)

    # # Wykres prognoz vs rzeczywiste wartości dla okresu niespokojnego (2023, pełny zestaw, testowy)
    # plt.figure(figsize=(12, 6))
    # plt.plot(dates_all[2], y_tests[2], label='Rzeczywiste ceny', color='blue')
    # plt.plot(dates_all[2], y_preds[2], label=f'Prognozy MLP (Keras, pełny zestaw, architektura {architecture})', color='red', linestyle='--')
    # plt.title('Prognozy cen energii (MLP Keras, okres niespokojny 2023, pełny zestaw, zbiór testowy, architektura {architecture})')
    # plt.xlabel('Data')
    # plt.ylabel('Cena (PLN/MWh)')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f"../../plots/prognozy_mlp_keras_niespokojny_full_test_{architecture}.png", dpi=300)
    # plt.close()

    # # Wykres prognoz vs rzeczywiste wartości dla okresu niespokojnego (2023, skrócony zestaw, testowy)
    # plt.figure(figsize=(12, 6))
    # plt.plot(dates_all[3], y_tests[3], label='Rzeczywiste ceny', color='blue')
    # plt.plot(dates_all[3], y_preds[3], label=f'Prognozy MLP (Keras, skrócony zestaw, architektura {architecture})', color='red', linestyle='--')
    # plt.title(f'Prognozy cen energii (MLP Keras, okres niespokojny 2023, skrócony zestaw, zbiór testowy,
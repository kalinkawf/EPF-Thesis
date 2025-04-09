import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Funkcja do obliczania MAPE (z poprawką na wartości bliskie 0)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = (y_true > 1) & (y_true != 0)  # Pomijamy wartości <= 1 i ujemne
    if not mask.any():
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Funkcja do obliczania sMAPE
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    mask = denominator != 0
    return np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100

# Funkcja do obliczania i wyświetlania metryk
def print_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name}:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Symmetric Mean Absolute Percentage Error (sMAPE): {smape:.2f}%")
    print(f"R²: {r2:.2f}")
    return mse, rmse, mae, mape, smape, r2

# Wczytaj dane
df = pd.read_csv("C:/mgr/EPF-Thesis/data/database.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Wybierz zmienne do użycia jako regresory w Prophecie
regressors = [
    "temp_waw", "wind_speed_waw", "cloud_cover_waw", "solar_radiation_waw",
    "temp_ksz", "wind_speed_ksz", "cloud_cover_ksz", "solar_radiation_ksz",
    "temp_krk", "wind_speed_krk", "cloud_cover_krk", "solar_radiation_krk",
    "temp_bab", "wind_speed_bab", "cloud_cover_bab", "solar_radiation_bab",
    "power_loss", "Network_loss",
    "Niemcy Bilans", "Czechy Bilans", "Litwa Bilans", "Słowacja Bilans", "Szwecja Bilans", "Ukraina Bilans",
    "hard_coal", "coal-derived", "lignite", "gas", "oil", "biomass", "wind_onshore", "solar",
    "fixing_i_volume", "Load", "gas_price", "gas_volume", "coal_pscmi1_pln_per_gj", "co2_price",
    "pln_usd", "brent_price", "day_of_week", "is_holiday"
]
target = "fixing_i_price"

# Usuń rekordy z NaN
df = df[regressors + [target] + ["timestamp"]].dropna()

# Przygotuj dane w formacie wymaganym przez Prophet (ds i y)
df_prophet = df.rename(columns={"timestamp": "ds", "fixing_i_price": "y"})

# Stwórz DataFrame z dniami świątecznymi na podstawie kolumny is_holiday
holidays = df_prophet[df_prophet["is_holiday"] == 1][["ds"]].drop_duplicates()
holidays["holiday"] = "pl_holidays"
holidays["lower_window"] = 0  # Efekt zaczyna się w dniu święta
holidays["upper_window"] = 1  # Efekt trwa 1 dzień po święcie
print("Liczba dni świątecznych:", len(holidays))

# --- Podział losowy (80/20) ---
train_size = int(len(df_prophet) * 0.8)
df_train_random = df_prophet.sample(frac=0.8, random_state=42)
df_test_random = df_prophet.drop(df_train_random.index).reset_index(drop=True)  # Reset indeksu

# Wyświetl zakresy dat dla podziału losowego
print("Podział losowy:")
print("Zakres dat dla zbioru treningowego (losowy):")
print(f"Od: {df_train_random['ds'].min()} do: {df_train_random['ds'].max()}")
print("\nZakres dat dla zbioru testowego (losowy):")
print(f"Od: {df_test_random['ds'].min()} do: {df_test_random['ds'].max()}")

# Model Prophet dla podziału losowego
model_prophet_random = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    holidays=holidays,  # Dodaj święta
    changepoint_prior_scale=0.1,
    seasonality_mode='multiplicative'
)

# Dodaj regresory (bez is_holiday, bo jest już w holidays)
regressors_without_holiday = [reg for reg in regressors if reg != "is_holiday"]
for regressor in regressors_without_holiday:
    model_prophet_random.add_regressor(regressor)

# Dopasuj model
model_prophet_random.fit(df_train_random)

# Przewidywanie
future_random = df_test_random.drop(columns=["y"]).reset_index(drop=True)
forecast_random = model_prophet_random.predict(future_random)
y_pred_random = forecast_random["yhat"]
y_test_random = df_test_random["y"]

# Oblicz metryki dla podziału losowego
metrics_random = print_metrics(y_test_random, y_pred_random, "Prophet (podział losowy)")

# Wykres dla podziału losowego (wszystkie rekordy)
plt.figure(figsize=(15, 6))
plt.plot(df_test_random["ds"], y_test_random, label="Rzeczywiste ceny", color="#3498db", alpha=0.7)
plt.plot(df_test_random["ds"], y_pred_random, label="Przewidywane ceny (Prophet, losowy)", color="#e74c3c", linestyle="--", alpha=0.7)
plt.title("Prophet (podział losowy): Rzeczywiste vs Przewidywane ceny (wszystkie rekordy)", fontsize=16, pad=20)
plt.xlabel("Data", fontsize=12)
plt.ylabel("Cena (PLN/MWh)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("C:/mgr/EPF-Thesis/plots/prophet_random_all_records_with_holidays.png", dpi=300)
plt.close()

# Zapisz wyniki (podział losowy)
results_random = pd.DataFrame({
    "Timestamp": df_test_random["ds"].reset_index(drop=True),
    "Rzeczywiste": y_test_random.reset_index(drop=True),
    "Przewidywane": y_pred_random.reset_index(drop=True)
})
results_random.to_csv("C:/mgr/EPF-Thesis/results/prophet_random_results_with_holidays.csv", index=False)

# --- Podział chronologiczny (trening 2016–2020, test 2021–2024) ---
# Sortuj dane według timestamp
df_sorted = df_prophet.sort_values(by="ds")

# Podział chronologiczny: trening 2016–2020, test 2021–2024
train_end_date = pd.to_datetime("2020-12-31 23:59:59")
df_train_chrono = df_sorted[df_sorted["ds"] <= train_end_date].reset_index(drop=True)
df_test_chrono = df_sorted[df_sorted["ds"] > train_end_date].reset_index(drop=True)

# Wyświetl zakresy dat dla podziału chronologicznego
print("\nPodział chronologiczny:")
print("Zakres dat dla zbioru treningowego (chronologiczny):")
print(f"Od: {df_train_chrono['ds'].min()} do: {df_train_chrono['ds'].max()}")
print("\nZakres dat dla zbioru testowego (chronologiczny):")
print(f"Od: {df_test_chrono['ds'].min()} do: {df_test_chrono['ds'].max()}")

# Model Prophet dla podziału chronologicznego
model_prophet_chrono = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    holidays=holidays,  # Dodaj święta
    changepoint_prior_scale=0.1,
    seasonality_mode='multiplicative'
)

# Dodaj regresory (bez is_holiday)
for regressor in regressors_without_holiday:
    model_prophet_chrono.add_regressor(regressor)

# Dopasuj model
model_prophet_chrono.fit(df_train_chrono)

# Przewidywanie
future_chrono = df_test_chrono.drop(columns=["y"]).reset_index(drop=True)
forecast_chrono = model_prophet_chrono.predict(future_chrono)
y_pred_chrono = forecast_chrono["yhat"]
y_test_chrono = df_test_chrono["y"]

# Oblicz metryki dla podziału chronologicznego
metrics_chrono = print_metrics(y_test_chrono, y_pred_chrono, "Prophet (podział chronologiczny)")

# Wykres dla podziału chronologicznego (wszystkie rekordy)
plt.figure(figsize=(15, 6))
plt.plot(df_test_chrono["ds"], y_test_chrono, label="Rzeczywiste ceny", color="#3498db", alpha=0.7)
plt.plot(df_test_chrono["ds"], y_pred_chrono, label="Przewidywane ceny (Prophet, chronologiczny)", color="#e74c3c", linestyle="--", alpha=0.7)
plt.title("Prophet (podział chronologiczny): Rzeczywiste vs Przewidywane ceny (wszystkie rekordy)", fontsize=16, pad=20)
plt.xlabel("Data", fontsize=12)
plt.ylabel("Cena (PLN/MWh)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("C:/mgr/EPF-Thesis/plots/prophet_chronological_all_records_with_holidays.png", dpi=300)
plt.close()

# Zapisz wyniki (podział chronologiczny)
results_chrono = pd.DataFrame({
    "Timestamp": df_test_chrono["ds"].reset_index(drop=True),
    "Rzeczywiste": y_test_chrono.reset_index(drop=True),
    "Przewidywane": y_pred_chrono.reset_index(drop=True)
})
results_chrono.to_csv("C:/mgr/EPF-Thesis/results/prophet_chronological_results_with_holidays.csv", index=False)

print("\nWyniki zapisane w C:/mgr/EPF-Thesis/results/")
print("Wykresy zapisane w C:/mgr/EPF-Thesis/plots/")
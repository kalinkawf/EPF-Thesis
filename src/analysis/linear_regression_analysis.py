import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Funkcja do obliczania MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Funkcja do obliczania i wyświetlania metryk
def print_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name}:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R²: {r2:.2f}")
    return mse, rmse, mae, mape, r2

# Wczytaj dane
df = pd.read_csv("C:/mgr/EPF-Thesis/data/database.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Wybierz wszystkie zmienne numeryczne (oprócz timestamp)
features = [
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
df = df[features + [target] + ["timestamp"]].dropna()

# Skalowanie danych za pomocą RobustScaler
scaler = RobustScaler()
X = scaler.fit_transform(df[features])
y = df[target]
timestamps = df["timestamp"]

# --- Podział losowy (80/20) ---
X_train_random, X_test_random, y_train_random, y_test_random, timestamps_train_random, timestamps_test_random = train_test_split(
    X, y, timestamps, test_size=0.2, random_state=42
)

# Wyświetl zakresy dat dla podziału losowego
print("Podział losowy:")
print("Zakres dat dla zbioru treningowego (losowy):")
print(f"Od: {timestamps_train_random.min()} do: {timestamps_train_random.max()}")
print("\nZakres dat dla zbioru testowego (losowy):")
print(f"Od: {timestamps_test_random.min()} do: {timestamps_test_random.max()}")

# Model Ridge dla podziału losowego
model_ridge_random = Ridge(alpha=1.0)
model_ridge_random.fit(X_train_random, y_train_random)
y_pred_random = model_ridge_random.predict(X_test_random)

# Oblicz metryki dla podziału losowego
metrics_random = print_metrics(y_test_random, y_pred_random, "Ridge Regression (podział losowy)")

# Wykres dla podziału losowego (wszystkie rekordy)
plt.figure(figsize=(15, 6))
plt.plot(timestamps_test_random, y_test_random, label="Rzeczywiste ceny", color="#3498db", alpha=0.7)
plt.plot(timestamps_test_random, y_pred_random, label="Przewidywane ceny (Ridge, losowy)", color="#e74c3c", linestyle="--", alpha=0.7)
plt.title("Ridge Regression (podział losowy): Rzeczywiste vs Przewidywane ceny (wszystkie rekordy)", fontsize=16, pad=20)
plt.xlabel("Data", fontsize=12)
plt.ylabel("Cena (PLN/MWh)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("C:/mgr/EPF-Thesis/plots/ridge_regression_random_all_records.png", dpi=300)
plt.close()

# --- Podział chronologiczny (trening 2016–2020, test 2021–2024) ---
# Sortuj dane według timestamp
df_sorted = df.sort_values(by="timestamp")
X_sorted = scaler.transform(df_sorted[features])
y_sorted = df_sorted[target]
timestamps_sorted = df_sorted["timestamp"]

# Podział chronologiczny: trening 2016–2020, test 2021–2024
train_end_date = pd.to_datetime("2020-12-31 23:59:59")
train_mask = timestamps_sorted <= train_end_date
test_mask = timestamps_sorted > train_end_date

X_train_chrono = X_sorted[train_mask]
X_test_chrono = X_sorted[test_mask]
y_train_chrono = y_sorted[train_mask]
y_test_chrono = y_sorted[test_mask]
timestamps_train_chrono = timestamps_sorted[train_mask]
timestamps_test_chrono = timestamps_sorted[test_mask]

# Wyświetl zakresy dat dla podziału chronologicznego
print("\nPodział chronologiczny:")
print("Zakres dat dla zbioru treningowego (chronologiczny):")
print(f"Od: {timestamps_train_chrono.min()} do: {timestamps_train_chrono.max()}")
print("\nZakres dat dla zbioru testowego (chronologiczny):")
print(f"Od: {timestamps_test_chrono.min()} do: {timestamps_test_chrono.max()}")

# Model Ridge dla podziału chronologicznego
model_ridge_chrono = Ridge(alpha=1.0)
model_ridge_chrono.fit(X_train_chrono, y_train_chrono)
y_pred_chrono = model_ridge_chrono.predict(X_test_chrono)

# Oblicz metryki dla podziału chronologicznego
metrics_chrono = print_metrics(y_test_chrono, y_pred_chrono, "Ridge Regression (podział chronologiczny)")

# Wykres dla podziału chronologicznego (wszystkie rekordy)
plt.figure(figsize=(15, 6))
plt.plot(timestamps_test_chrono, y_test_chrono, label="Rzeczywiste ceny", color="#3498db", alpha=0.7)
plt.plot(timestamps_test_chrono, y_pred_chrono, label="Przewidywane ceny (Ridge, chronologiczny)", color="#e74c3c", linestyle="--", alpha=0.7)
plt.title("Ridge Regression (podział chronologiczny): Rzeczywiste vs Przewidywane ceny (wszystkie rekordy)", fontsize=16, pad=20)
plt.xlabel("Data", fontsize=12)
plt.ylabel("Cena (PLN/MWh)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("C:/mgr/EPF-Thesis/plots/ridge_regression_chronological_all_records.png", dpi=300)
plt.close()

# Współczynniki modelu (Ridge) dla podziału chronologicznego
coefficients = pd.DataFrame(model_ridge_chrono.coef_, index=features, columns=["Współczynnik"])
print("\nWspółczynniki modelu (Ridge, podział chronologiczny):")
print(coefficients)

# Zapisz wyniki (podział chronologiczny)
results_chrono = pd.DataFrame({
    "Timestamp": timestamps_test_chrono,
    "Rzeczywiste": y_test_chrono.values,
    "Przewidywane": y_pred_chrono
})
results_chrono.to_csv("C:/mgr/EPF-Thesis/results/ridge_regression_chronological_results.csv", index=False)

# Zapisz wyniki (podział losowy)
results_random = pd.DataFrame({
    "Timestamp": timestamps_test_random,
    "Rzeczywiste": y_test_random.values,
    "Przewidywane": y_pred_random
})
results_random.to_csv("C:/mgr/EPF-Thesis/results/ridge_regression_random_results.csv", index=False)

print("\nWyniki zapisane w C:/mgr/EPF-Thesis/results/")
print("Wykresy zapisane w C:/mgr/EPF-Thesis/plots/")
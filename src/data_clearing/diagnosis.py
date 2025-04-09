import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# Wczytaj dane
df = pd.read_csv("C:/mgr/EPF-Thesis/data/database.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# --- Test 1: Sprawdzenie rozkładu fixing_i_price ---
print("Statystyki opisowe dla fixing_i_price (przed usunięciem NaN):")
print(df["fixing_i_price"].describe())
print("\nLiczba wartości ujemnych:", (df["fixing_i_price"] < 0).sum())
print("Liczba wartości <= 1:", (df["fixing_i_price"] <= 1).sum())
print("Liczba wartości == 0:", (df["fixing_i_price"] == 0).sum())

# Histogram fixing_i_price
plt.figure(figsize=(10, 6))
plt.hist(df["fixing_i_price"].dropna(), bins=100, color="#3498db", alpha=0.7)
plt.title("Histogram fixing_i_price", fontsize=16)
plt.xlabel("Cena (PLN/MWh)", fontsize=12)
plt.ylabel("Liczba rekordów", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("C:/mgr/EPF-Thesis/plots/fixing_i_price_histogram.png", dpi=300)
plt.close()

# --- Test 2: Sprawdzenie brakujących wartości (NaN) ---
print("\nLiczba brakujących wartości (NaN) w fixing_i_price:", df["fixing_i_price"].isna().sum())

# Usuń rekordy z NaN
df_clean = df.dropna(subset=["fixing_i_price"])
print("Statystyki opisowe dla fixing_i_price (po usunięciu NaN):")
print(df_clean["fixing_i_price"].describe())

# --- Test 3: Analiza błędów procentowych (na przykładzie Propheta, podział losowy) ---
# Przygotuj dane w formacie wymaganym przez Prophet
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
df_clean = df_clean[regressors + [target] + ["timestamp"]].dropna()

# Przygotuj dane dla Propheta
df_prophet = df_clean.rename(columns={"timestamp": "ds", "fixing_i_price": "y"})

# Podział losowy (80/20)
train_size = int(len(df_prophet) * 0.8)
df_train_random = df_prophet.sample(frac=0.8, random_state=42)
df_test_random = df_prophet.drop(df_train_random.index)

# Wczytaj wyniki Propheta (z poprzedniego kodu)
results_random = pd.read_csv("C:/mgr/EPF-Thesis/results/prophet_random_results.csv")
results_random["Timestamp"] = pd.to_datetime(results_random["Timestamp"])

# Oblicz indywidualne błędy procentowe
results_random["Absolute_Percentage_Error"] = np.abs((results_random["Rzeczywiste"] - results_random["Przewidywane"]) / results_random["Rzeczywiste"]) * 100

# Wyświetl rekordy z największymi błędami procentowymi
print("\nRekordy z największymi błędami procentowymi (Top 10):")
print(results_random.nlargest(10, "Absolute_Percentage_Error")[["Timestamp", "Rzeczywiste", "Przewidywane", "Absolute_Percentage_Error"]])

# Histogram błędów procentowych
plt.figure(figsize=(10, 6))
plt.hist(results_random["Absolute_Percentage_Error"], bins=100, color="#e74c3c", alpha=0.7)
plt.title("Histogram błędów procentowych (Prophet, podział losowy)", fontsize=16)
plt.xlabel("Błąd procentowy (%)", fontsize=12)
plt.ylabel("Liczba rekordów", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("C:/mgr/EPF-Thesis/plots/absolute_percentage_error_histogram.png", dpi=300)
plt.close()

# --- Test 4: Oblicz sMAPE dla Propheta (podział losowy) ---
y_test_random = results_random["Rzeczywiste"]
y_pred_random = results_random["Przewidywane"]
smape_random = symmetric_mean_absolute_percentage_error(y_test_random, y_pred_random)
print("\nsMAPE dla Propheta (podział losowy):", smape_random, "%")

# --- Test 5: Sprawdzenie ujemnych przewidywań ---
print("\nLiczba ujemnych przewidywań (Prophet, podział losowy):", (results_random["Przewidywane"] < 0).sum())
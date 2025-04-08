import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Krok 1: Przygotowanie danych
# Wczytaj dane
df = pd.read_csv("C:/mgr/EPF-Thesis/data/combined_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Wybierz zmienne niezależne (features) i zmienną zależną (target)
# Na podstawie wcześniejszej analizy korelacji wybieramy zmienne o wysokim wpływie
features = [
    "Load", "wind_onshore", "solar", "gas_price", "co2_price",
    "day_of_week", "is_holiday", "fixing_i_volume", "brent_price"
]
target = "fixing_i_price"

# Usuń rekordy z NaN (w Twoim przypadku nie ma NaN-ów, ale dla pewności)
df = df[features + [target]].dropna()

# Normalizacja danych
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
y = df[target]

# Podziel dane na zbiór treningowy i testowy (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Krok 2: Implementacja modelu regresji liniowej
model = LinearRegression()
model.fit(X_train, y_train)

# Krok 3: Przewidywanie i ocena modelu
y_pred = model.predict(X_test)

# Oblicz metryki
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Regresja liniowa - Mean Squared Error (MSE): {mse:.2f}")
print(f"Regresja liniowa - Mean Absolute Error (MAE): {mae:.2f}")

# Współczynniki modelu
coefficients = pd.DataFrame(model.coef_, index=features, columns=["Współczynnik"])
print("\nWspółczynniki modelu:")
print(coefficients)

# Wykres porównujący rzeczywiste i przewidywane ceny (pierwsze 100 rekordów dla czytelności)
plt.style.use("seaborn")
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:100], label="Rzeczywiste ceny", color="#3498db")
plt.plot(y_pred[:100], label="Przewidywane ceny", color="#e74c3c", linestyle="--")
plt.title("Regresja liniowa: Rzeczywiste vs Przewidywane ceny (pierwsze 100 rekordów)", fontsize=16, pad=20)
plt.xlabel("Indeks", fontsize=12)
plt.ylabel("Cena (PLN/MWh)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("C:/mgr/EPF-Thesis/plots/linear_regression_predictions.png", dpi=300)
plt.close()

# Zapisz wyniki do pliku
results = pd.DataFrame({
    "Rzeczywiste": y_test.values,
    "Przewidywane": y_pred
}, index=y_test.index)
results.to_csv("C:/mgr/EPF-Thesis/results/linear_regression_results.csv")

print("Wyniki zapisane w C:/mgr/EPF-Thesis/results/linear_regression_results.csv")
print("Wykres zapisany w C:/mgr/EPF-Thesis/plots/linear_regression_predictions.png")
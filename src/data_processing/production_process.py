import pandas as pd
import numpy as np

# Ścieżki
input_file = "../../data/other_data/electricity_production_entsoe_all.csv"  # Zmień na nazwę swojego pliku
output_file = "../../data/energy_sources_prod.csv"

# Lista kolumn, które chcesz zachować
columns_to_keep = [
    "date",
    "hard_coal",
    "coal-derived",
    "lignite",
    "gas",
    "oil",
    "biomass",
    "wind_onshore",
    "solar"
]

# Wczytaj dane
print(f"Wczytuję dane z pliku: {input_file}")
df = pd.read_csv(input_file)

# Sprawdź, czy wszystkie wymagane kolumny istnieją w pliku
missing_columns = [col for col in columns_to_keep if col not in df.columns]
if missing_columns:
    print(f"Brakujące kolumny w pliku: {missing_columns}")
    exit()

# Wybierz tylko określone kolumny
df = df[columns_to_keep]

columns_to_convert = [
    "hard_coal",
    "coal-derived",
    "lignite",
    "gas",
    "oil",
    "biomass",
    "wind_onshore",
    "solar"
]

# Konwertuj date na datetime z odpowiednim formatem
df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y %H:%M")

# Zaokrąglij date do pełnych godzin i uśrednij wartości
df["hourly_timestamp"] = df["date"].dt.floor("h")
df_hourly = df.groupby("hourly_timestamp").agg({
    "hard_coal": "mean",
    "coal-derived": "mean",
    "lignite": "mean",
    "gas": "mean",
    "oil": "mean",
    "biomass": "mean",
    "wind_onshore": "mean",
    "solar": "mean"
}).reset_index()

# Zmień nazwę kolumny hourly_timestamp na date
df_hourly = df_hourly.rename(columns={"hourly_timestamp": "date"})

df_hourly["solar"] = df["solar"].fillna(0.0)

# Zapisz do nowego pliku CSV
df_hourly.to_csv(output_file, index=False, encoding="utf-8")
print(f"Dane godzinowe (w MW) zapisane do pliku: {output_file}")
print(f"Liczba rekordów: {len(df_hourly)}")
print(f"Kolumny w pliku: {df_hourly.columns.tolist()}")

# Wyświetl pierwsze 5 wierszy
print("\nPierwsze 5 wierszy danych godzinowych (w MW):")
print(df_hourly.head())
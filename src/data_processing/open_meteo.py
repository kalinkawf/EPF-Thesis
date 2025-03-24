import pandas as pd
import os

# Ścieżki
input_dir = "../../data/weather_meteo/"
output_file = "../../data/weather_hourly.csv"

# Lista lokalizacji i odpowiadających im sufiksów
locations = {
    "waw": "waw",    # Warszawa
    "ksz": "ksz",    # Koszalin
    "krk": "krk",    # Kraków
    "bab": "bab"     # Babimost
}

# Parametry do wybrania i ich nowe nazwy
parameters = {
    "temperature_2m (°C)": "temp",
    "wind_speed_100m (m/s)": "wind_speed",
    "cloud_cover (%)": "cloud_cover",
    "shortwave_radiation (W/m²)": "solar_radiation"
}

# Wczytaj dane dla każdej lokalizacji
all_data = []
for location, suffix in locations.items():
    file_path = os.path.join(input_dir, f"open-meteo-{location}.csv")
    print(f"Wczytuję dane dla {location}: {file_path}")
    
    # Wczytaj plik
    df = pd.read_csv(file_path)
    
    # Konwertuj kolumnę time na datetime i zmień format
    df["timestamp"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Usuń oryginalną kolumnę time
    df = df.drop(columns=["time"])
    
    # Zmień nazwy kolumn
    selected_cols = {"timestamp": "timestamp"}
    for param, prefix in parameters.items():
        if param in df.columns:
            selected_cols[param] = f"{prefix}_{suffix}"
    
    df_selected = df.rename(columns=selected_cols)
    
    # Przelicz zachmurzenie z % (0–100) na skalę oktantową (0–8)
    cloud_cover_col = f"cloud_cover_{suffix}"
    if cloud_cover_col in df_selected.columns:
        df_selected[cloud_cover_col] = df_selected[cloud_cover_col] * 8 / 100
    
    all_data.append(df_selected)

# Połącz dane w jeden DataFrame
df_combined = all_data[0]
for df in all_data[1:]:
    df_combined = df_combined.merge(df, on="timestamp", how="outer")

# Posortuj według timestamp
df_combined["timestamp"] = pd.to_datetime(df_combined["timestamp"])
df_combined = df_combined.sort_values("timestamp")

# Zapisz do nowego pliku CSV
df_combined.to_csv(output_file, index=False, encoding="utf-8")
print(f"Dane pogodowe zapisane do pliku: {output_file}")
print(f"Liczba rekordów: {len(df_combined)}")
print(f"Kolumny w pliku: {df_combined.columns.tolist()}")

# Wyświetl pierwsze 5 wierszy
print("\nPierwsze 5 wierszy danych:")
print(df_combined.head())

# Sprawdź, ile jest brakujących wartości w każdej kolumnie
print("\nLiczba brakujących wartości (NaN) w każdej kolumnie:")
print(df_combined.isna().sum())
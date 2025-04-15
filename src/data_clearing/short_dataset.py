# Importowanie bibliotek
import pandas as pd

# Wczytanie danych
df = pd.read_csv("../../data/database.csv")

# Konwersja timestamp na format datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Konwersja zmiennych logicznych na numeryczne
df["is_holiday"] = df["is_holiday"].astype(int)

# Obliczenie średnich dla zmiennych pogodowych
df["avg_temp"] = df[["temp_waw", "temp_ksz", "temp_krk", "temp_bab"]].mean(axis=1)
df["avg_wind_speed"] = df[["wind_speed_waw", "wind_speed_ksz", "wind_speed_krk", "wind_speed_bab"]].mean(axis=1)
df["avg_solar_radiation"] = df[["solar_radiation_waw", "solar_radiation_ksz", "solar_radiation_krk", "solar_radiation_bab"]].mean(axis=1)

# Definicja nowego skróconego zestawu danych
columns_to_keep = [
    "timestamp", "fixing_i_price",  # Zmienna czasowa i zmienna celu
    "fixing_i_price_lag24", "fixing_i_price_lag168",
    "gas_price", "co2_price", "brent_price", "pln_usd", "coal_pscmi1_pln_per_gj",
    "power_loss", "fixing_i_volume", "solar", "gas", "oil", "Load",
    "avg_temp", "avg_wind_speed", "avg_solar_radiation",
    "hour", "month", "is_holiday", "wind_onshore", "day_of_week"
]

# Utworzenie nowego DataFrame z wybranymi kolumnami
df_short = df[columns_to_keep].copy()

# Usunięcie brakujących wartości (jeśli istnieją)
df_short = df_short.dropna()

# Zapisanie nowego skróconego zestawu danych do pliku short_database.csv
df_short.to_csv("../../data/short_database.csv", index=False)

# Wyświetlenie informacji o nowym zestawie danych
print("Nowy skrócony zestaw danych został zapisany jako ../../data/short_database.csv")
print(f"Liczba kolumn: {len(df_short.columns)}")
print(f"Kolumny w nowym zestawie danych: {list(df_short.columns)}")
print(f"Liczba wierszy: {len(df_short)}")
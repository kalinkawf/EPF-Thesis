import pandas as pd
import os

# Ścieżki do plików
input_dir = "../../data/"
output_file = "../../data/combined_data.csv"

# Lista plików i ich kolumn z czasem
files = {
    "weather_hourly.csv": "Time",
    "power_outages.csv": "Time",
    "import_export.csv": "Time",
    "energy_sources_prod.csv": "date",
    "electricity_prices_day_ahead_hourly_all.csv": "date"
}

# Wczytaj dane i ujednolić timestampy
all_data = []
for file_name, time_col in files.items():
    file_path = os.path.join(input_dir, file_name)
    print(f"Wczytuję dane z pliku: {file_path}")
    
    # Wczytaj plik
    df = pd.read_csv(file_path)
    
    # Specjalne parsowanie dla electricity_prices_day_ahead_hourly_all.csv
    if file_name == "electricity_prices_day_ahead_hourly_all.csv":
        # Parsuj datę w formacie DD.MM.YYYY HH:MM
        df["timestamp"] = pd.to_datetime(df[time_col], format="%d.%m.%Y %H:%M")
        df = df.drop(columns='fixing_ii_price')
        df = df.drop(columns='fixing_ii_volume')
    else:
        # Dla pozostałych plików zakładamy standardowy format YYYY-MM-DD HH:MM:SS
        df["timestamp"] = pd.to_datetime(df[time_col])
    
    # Usuń oryginalną kolumnę z czasem
    df = df.drop(columns=[time_col])
    
    all_data.append(df)

# Stwórz wspólny zakres czasowy (godzinowy)
min_timestamp = min(df["timestamp"].min() for df in all_data)
max_timestamp = max(df["timestamp"].max() for df in all_data)
common_timestamps = pd.date_range(start=min_timestamp, end=max_timestamp, freq="H")
common_df = pd.DataFrame({"timestamp": common_timestamps})

# Połącz dane w jeden DataFrame
combined_df = common_df
for df in all_data:
    combined_df = combined_df.merge(df, on="timestamp", how="outer")

# Posortuj według timestamp
combined_df = combined_df.sort_values("timestamp")

# Usuń rekordy, gdzie timestamp ma niezerowe minuty
combined_df = combined_df[combined_df['timestamp'].dt.minute == 0]

# Usuń rekordy, gdzie jakakolwiek kolumna zawiera NaN
combined_df = combined_df.dropna()

# Weryfikacja liczby rekordów
print(f"\nMinimalny timestamp: {combined_df['timestamp'].min()}")
print(f"Maksymalny timestamp: {combined_df['timestamp'].max()}")

# Oblicz oczekiwaną liczbę rekordów
expected_records = (combined_df['timestamp'].max() - combined_df['timestamp'].min()).total_seconds() / 3600 + 1
print(f"Oczekiwana liczba rekordów: {int(expected_records)}")
print(f"Rzeczywista liczba rekordów: {len(combined_df)}")

# Sprawdź duplikaty timestampów
duplicates = combined_df['timestamp'].duplicated().sum()
print(f"Liczba zduplikowanych timestampów: {duplicates}")
if duplicates > 0:
    print("Przykładowe zduplikowane timestampy:")
    print(combined_df[combined_df['timestamp'].duplicated(keep=False)].head())

# Sprawdź, czy różnice między timestampami wynoszą dokładnie 1 godzinę
combined_df['time_diff'] = combined_df['timestamp'].diff().dt.total_seconds() / 3600
incorrect_diffs = combined_df[combined_df['time_diff'].notna() & (combined_df['time_diff'] != 1.0)]
if not incorrect_diffs.empty:
    print(f"\nZnaleziono {len(incorrect_diffs)} timestampów z nieprawidłowymi różnicami czasu:")
    print(incorrect_diffs[['timestamp', 'time_diff']].head())
else:
    print("\nWszystkie różnice między timestampami wynoszą dokładnie 1 godzinę.")

# Usuń tymczasową kolumnę time_diff
combined_df = combined_df.drop(columns=['time_diff'])

# Wyświetl informacje o brakujących wartościach
print("\nLiczba brakujących wartości (NaN) w każdej kolumnie:")
print(combined_df.isna().sum())

# Zapisz do nowego pliku CSV
combined_df.to_csv(output_file, index=False, encoding="utf-8")
print(f"\nDane połączone zapisane do pliku: {output_file}")
print(f"Liczba rekordów: {len(combined_df)}")
print(f"Kolumny w pliku: {combined_df.columns.tolist()}")

# Wyświetl pierwsze 5 wierszy
print("\nPierwsze 5 wierszy danych:")
print(combined_df.head())
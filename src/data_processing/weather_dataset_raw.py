import pandas as pd
import os
from pathlib import Path

# Ścieżki
base_path = "../dane/pogoda/"
output_file = "../dane/pogoda/weather_dataset_raw.csv"

# Problemy z centralną Polską 
# 252200230 - Warszawa-Filtry - troche mniej danych 9681
# 252200150 - Warszawa-Bielany - malo danych 8306 bardzo źle
# 251200030 - Skierniewice mało 9771 kompletne
# 252210050 - Pułtusk slabo 9767 niekompletne
# Wybrane stacje
selected_stations = ["251200030", "254180090", "250190390", "252150270"]

# Nazwy kolumn (na podstawie opisu)
columns = [
    "station_code", "station_name", "year", "month", "day", "hour",
    "temp", "temp_status", "temp_wet", "temp_wet_status", "ice_indicator", "ventilation_indicator",
    "humidity", "humidity_status", "wind_direction", "wind_direction_status",
    "wind_speed", "wind_speed_status", "cloud_cover", "cloud_cover_status",
    "visibility", "visibility_status"
]

# Lista do przechowywania danych
all_data = []

# Przetwarzanie danych dla każdego roku i miesiąca
for year in range(2016, 2025):
    year_folder = os.path.join(base_path, str(year))
    if not os.path.exists(year_folder):
        print(f"Folder {year_folder} nie istnieje, pomijam.")
        continue

    for month in range(1, 13):
        month_str = f"{month:02d}"
        file_name = f"k_t_{month_str}_{year}.csv"
        file_path = os.path.join(year_folder, file_name)

        if not os.path.exists(file_path):
            print(f"Plik {file_path} nie istnieje, pomijam.")
            continue

        print(f"Przetwarzam plik: {file_path}")

        try:
            # Wczytaj plik CSV bez nagłówków
            df = pd.read_csv(file_path, encoding="cp1250", header=None, names=columns)
        except Exception as e:
            print(f"Błąd wczytywania pliku {file_path}: {e}")
            continue

        # Wybierz tylko wybrane stacje
        df = df[df["station_code"].astype(str).isin(selected_stations)]

        if df.empty:
            print(f"Brak danych dla wybranych stacji w pliku {file_path}, pomijam.")
            continue

        # Oznacz braki (status = 8)
        for col, status_col in [
            ("temp", "temp_status"),
            ("wind_speed", "wind_speed_status"),
            ("cloud_cover", "cloud_cover_status")
        ]:
            df.loc[df[status_col] == 8, col] = pd.NA

        # Wybierz potrzebne kolumny
        df = df[["year", "month", "day", "hour", "station_code", "temp", "wind_speed", "cloud_cover"]]

        # Dodaj do listy
        all_data.append(df)

# Połącz wszystkie dane
if not all_data:
    raise ValueError("Brak danych dla wybranych stacji w podanych plikach.")

df_combined = pd.concat(all_data, ignore_index=True)

# Sortuj według year, month, day, hour i station_code
df_combined = df_combined.sort_values(["year", "month", "day", "hour", "station_code"])

# Zapisz do pliku CSV
df_combined.to_csv(output_file, index=False, encoding="utf-8")
print(f"Dataset zapisany jako: {output_file}")
print(f"Liczba rekordów: {len(df_combined)}")
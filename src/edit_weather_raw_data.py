import pandas as pd
from pathlib import Path

# Ścieżki
input_file = "../dane/pogoda/weather_dataset_raw.csv"
output_file = "../dane/pogoda/weather_dataset.csv"

# Mapowanie kodów stacji na nowe oznaczenia
station_mapping = {
    "251200030": "1",  # Skierniewice → Skr (region centralny)
    "254180090": "2",  # Gdańsk-Rębiechowo → Gda (region północny)
    "250190390": "3",  # Kraków-Obserwatorium → Krk (region południowy)
    "252150270": "4"   # Babimost → Babimost (region zachodni)
}

# Wczytaj dane
print(f"Wczytuję plik: {input_file}")
df = pd.read_csv(input_file)

# Utwórz timestamp
df["timestamp"] = pd.to_datetime(
    df[["year", "month", "day", "hour"]].astype(str).agg("-".join, axis=1),
    format="%Y-%m-%d-%H"
)

# Mapuj kody stacji na nowe oznaczenia (1, 2, 3, 4)
df["station_code"] = df["station_code"].astype(str).map(station_mapping)

# Sprawdź, czy są jakieś NaN w station_code (czyli stacje, które nie pasują do mapowania)
if df["station_code"].isna().any():
    print("Znaleziono rekordy z nieznanymi kodami stacji:")
    print(df[df["station_code"].isna()])
    # Usuń rekordy z nieznanymi kodami stacji (np. 250190390, Warszawa-Filtry)
    df = df.dropna(subset=["station_code"])

# Sprawdź kompletność danych dla każdej stacji
station_counts = df["station_code"].value_counts()
print("Liczba rekordów dla każdej stacji:")
print(station_counts)

# Sprawdź duplikaty
duplicates = df.duplicated(subset=["timestamp", "station_code"], keep=False)
if duplicates.any():
    print("Znaleziono zduplikowane wiersze w danych:")
    print(df[duplicates].sort_values(["timestamp", "station_code"]))
    print("Usuwam duplikaty, zachowując pierwszy wiersz...")
    df = df.drop_duplicates(subset=["timestamp", "station_code"], keep="first")

# Przekształć na format szeroki
df_pivot = df.pivot(index="timestamp", columns="station_code", values=["temp", "wind_speed", "cloud_cover"])

# Spłaszcz nazwy kolumn
df_pivot.columns = [f"{var}_{station}" for var, station in df_pivot.columns]
df_pivot.reset_index(inplace=True)

# Zmień nazwy kolumn na żądane (np. temp_1 → temp_waw)
df_pivot = df_pivot.rename(columns={
    "temp_1": "temp_skr",
    "temp_2": "temp_gda",
    "temp_3": "temp_krk",
    "temp_4": "temp_babimost",
    "wind_speed_1": "wind_speed_skr",
    "wind_speed_2": "wind_speed_gda",
    "wind_speed_3": "wind_speed_krk",
    "wind_speed_4": "wind_speed_babimost",
    "cloud_cover_1": "cloud_cover_skr",
    "cloud_cover_2": "cloud_cover_gda",
    "cloud_cover_3": "cloud_cover_krk",
    "cloud_cover_4": "cloud_cover_babimost"
})

# Utwórz pełny zakres timestamp (tylko godziny 6:00, 12:00, 18:00 od 2016-01-01 do 2024-12-31)
# Najpierw tworzymy pełny zakres dni
dates = pd.date_range(start="2016-01-01", end="2024-12-31", freq="D")
# Dla każdego dnia generujemy godziny 6:00, 12:00, 18:00
timestamps = []
for date in dates:
    for hour in [6, 12, 18]:
        timestamps.append(pd.Timestamp(year=date.year, month=date.month, day=date.day, hour=hour))
full_timestamps = pd.DatetimeIndex(timestamps)
full_df = pd.DataFrame({"timestamp": full_timestamps})

# Połącz z df_pivot, aby dodać brakujące timestamp
df_final = full_df.merge(df_pivot, on="timestamp", how="left")

# Upewnij się, że kolumny są w odpowiedniej kolejności
final_columns = [
    "timestamp",
    "temp_skr", "temp_krk", "temp_gda", "temp_babimost",
    "wind_speed_skr", "wind_speed_krk", "wind_speed_gda", "wind_speed_babimost",
    "cloud_cover_skr", "cloud_cover_krk", "cloud_cover_gda", "cloud_cover_babimost"
]

# Sprawdź, czy wszystkie kolumny istnieją, jeśli nie, utwórz je z wartościami NaN
for col in final_columns:
    if col not in df_final.columns:
        df_final[col] = pd.NA

# Wybierz finalne kolumny
df_final = df_final[final_columns]

# Interpolacja liniowa dla wartości NaN
numeric_columns = [col for col in final_columns if col != "timestamp"]
df_final[numeric_columns] = df_final[numeric_columns].interpolate(method="linear")

# Zapisz do pliku CSV
df_final.to_csv(output_file, index=False, encoding="utf-8")
print(f"Dataset zapisany jako: {output_file}")
print(f"Liczba rekordów: {len(df_final)}")
print(f"Kolumny w pliku: {df_final.columns.tolist()}")
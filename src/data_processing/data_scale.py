import pandas as pd
import os
from datetime import datetime, timedelta
import holidays as hl

# Ścieżki do plików
input_dir = "../../data/processed_data/"
other_data_input_dir = "../../data/other_data/"
output_dir = "../../data/processed_data/"
output_file = "../../data/database.csv"
os.makedirs(output_dir, exist_ok=True)

# Lista plików i ich kolumn z czasem
files = {
    "weather_hourly.csv": "Time",
    "power_outages.csv": "Time",
    "import_export.csv": "Time",
    "energy_sources_prod.csv": "date",
    "electricity_prices_day_ahead_hourly_all.csv": "date",
    "load.csv": "date",
    "non_emissive.csv": "Time",
    "rb_data.csv": "Time",
    "foreign_prices.csv": "Time",
}

price_files = {
    "prices_gas_day_ahead_all.csv": ["gas_price", "gas_volume"],
    "prices_coal_all.csv": ["coal_pscmi1_pln_per_gj"],
    "prices_eu_co2.csv": ["co2_price"],
    "usd_eur_pln_daily.csv": ["pln_usd", "pln_eur"],
    "Europe_Brent_Spot_Price.csv": ["Brent_USD"],
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
    if time_col in df.columns:
        df = df.drop(columns=[time_col])
    
    all_data.append(df)

# Stwórz wspólny zakres czasowy (godzinowy)
min_timestamp = pd.Timestamp("2016-01-01 00:00:00")
max_timestamp = pd.Timestamp("2023-12-31 23:00:00")
common_timestamps = pd.date_range(start=min_timestamp, end=max_timestamp, freq="h")
common_df = pd.DataFrame({"timestamp": common_timestamps})

# Połącz dane w jeden DataFrame
combined_df = common_df
for df in all_data:
    combined_df = combined_df.merge(df, on="timestamp", how="outer")

# Wczytaj dane z plików z cenami (dzienne, miesięczne i nieregularne)
all_new_data = []
for file_name, selected_cols in price_files.items():
    file_path = os.path.join(other_data_input_dir, file_name)
    print(f"\nWczytuję dane z pliku: {file_path}")
    
    # Wczytaj plik
    df = pd.read_csv(file_path)
    if "usd_eur" in file_name:
        # Parsuj datę w formacie YYYY-MM-DD
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    elif "Europe_Brent_Spot_Price" in file_name:
        # Parsuj datę w formacie DD.MM.YYYY
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    else:
        df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y", errors="coerce")
    
    # Wybierz tylko potrzebne kolumny
    df = df[["date"] + selected_cols]
    
    # Obsługa danych dziennych (ceny gazu)
    if "gas" in file_name:
        # Rozszerz dane dzienne na godzinowe
        hourly_data = []
        for _, row in df.iterrows():
            day = row["date"]
            for hour in range(24):
                timestamp = day + pd.Timedelta(hours=hour)
                new_row = {"timestamp": timestamp}
                for col in selected_cols:
                    new_row[col] = row[col]
                hourly_data.append(new_row)
        
        # Stwórz DataFrame z danymi godzinowymi
        hourly_df = pd.DataFrame(hourly_data)
        all_new_data.append(hourly_df)
    
    # Obsługa danych miesięcznych (ceny węgla)
    elif "coal" in file_name:
        # Rozszerz dane miesięczne na dzienne, a następnie na godzinowe
        hourly_data = []
        for _, row in df.iterrows():
            month_start = row["date"]
            # Znajdź koniec miesiąca
            if month_start.month == 12:
                month_end = month_start.replace(year=month_start.year + 1, month=1, day=1) - pd.Timedelta(days=1)
            else:
                month_end = month_start.replace(month=month_start.month + 1, day=1) - pd.Timedelta(days=1)
            
            # Wygeneruj wszystkie dni w danym miesiącu
            current_day = month_start
            while current_day <= month_end:
                # Dla każdego dnia wygeneruj 24 rekordy godzinowe
                for hour in range(24):
                    timestamp = current_day + pd.Timedelta(hours=hour)
                    new_row = {"timestamp": timestamp}
                    for col in selected_cols:
                        new_row[col] = row[col]
                    hourly_data.append(new_row)
                current_day += pd.Timedelta(days=1)
        
        # Stwórz DataFrame z danymi godzinowymi
        hourly_df = pd.DataFrame(hourly_data)
        all_new_data.append(hourly_df)
    
    # Obsługa danych nieregularnych (ceny CO2)
    elif "co2" in file_name:
        # Interpolacja liniowa na dane dzienne
        # Stwórz ciągły zakres czasowy (dzienny) od min do max daty
        min_date = df["date"].min()
        max_date = df["date"].max()
        daily_dates = pd.date_range(start=min_date, end=max_date, freq="D")
        daily_df = pd.DataFrame({"date": daily_dates})
        
        # Połącz z oryginalnymi danymi
        daily_df = daily_df.merge(df, on="date", how="left")
        
        # Interpolacja liniowa dla wybranych kolumn
        for col in selected_cols:
            daily_df[col] = daily_df[col].interpolate(method="linear")
        
        # Rozszerz dane dzienne na godzinowe
        hourly_data = []
        for _, row in daily_df.iterrows():
            day = row["date"]
            for hour in range(24):
                timestamp = day + pd.Timedelta(hours=hour)
                new_row = {"timestamp": timestamp}
                for col in selected_cols:
                    new_row[col] = row[col]
                hourly_data.append(new_row)
        
        # Stwórz DataFrame z danymi godzinowymi
        hourly_df = pd.DataFrame(hourly_data)
        all_new_data.append(hourly_df)
    
    # Obsługa danych nieregularnych (kursy walut z NBP)
    elif "usd_eur" in file_name:
        # Interpolacja liniowa na dane dzienne
        # Stwórz ciągły zakres czasowy (dzienny) od min do max daty
        min_date = df["date"].min()
        max_date = df["date"].max()
        daily_dates = pd.date_range(start=min_date, end=max_date, freq="D")
        daily_df = pd.DataFrame({"date": daily_dates})
        
        # Połącz z oryginalnymi danymi
        daily_df = daily_df.merge(df, on="date", how="left")
        
        # Interpolacja liniowa dla wybranych kolumn
        for col in selected_cols:
            daily_df[col] = daily_df[col].interpolate(method="linear")
            # Wypełnij wartości NaN na początku i końcu (jeśli istnieją) metodą forward/backward fill
            daily_df[col] = daily_df[col].ffill().bfill()
        
        # Rozszerz dane dzienne na godzinowe
        hourly_data = []
        for _, row in daily_df.iterrows():
            day = row["date"]
            for hour in range(24):
                timestamp = day + pd.Timedelta(hours=hour)
                new_row = {"timestamp": timestamp}
                for col in selected_cols:
                    new_row[col] = row[col]
                hourly_data.append(new_row)
        
        # Stwórz DataFrame z danymi godzinowymi
        hourly_df = pd.DataFrame(hourly_data)
        all_new_data.append(hourly_df)
    elif "Europe_Brent_Spot_Price" in file_name:
        # Interpolacja liniowa na dane dzienne
        # Stwórz ciągły zakres czasowy (dzienny) od min do max daty
        min_date = df["date"].min()
        max_date = df["date"].max()
        daily_dates = pd.date_range(start=min_date, end=max_date, freq="D")
        daily_df = pd.DataFrame({"date": daily_dates})
        
        # Połącz z oryginalnymi danymi
        daily_df = daily_df.merge(df, on="date", how="left")
        
        # Interpolacja liniowa dla wybranych kolumn
        for col in selected_cols:
            daily_df[col] = daily_df[col].interpolate(method="linear")
            # Wypełnij wartości NaN na początku i końcu (jeśli istnieją) metodą forward/backward fill
            daily_df[col] = daily_df[col].ffill().bfill()
        
        # Rozszerz dane dzienne na godzinowe
        hourly_data = []
        for _, row in daily_df.iterrows():
            day = row["date"]
            for hour in range(24):
                timestamp = day + pd.Timedelta(hours=hour)
                new_row = {"timestamp": timestamp}
                for col in selected_cols:
                    new_row[col] = row[col]
                hourly_data.append(new_row)
        
        # Stwórz DataFrame z danymi godzinowymi
        hourly_df = pd.DataFrame(hourly_data)
        all_new_data.append(hourly_df)

# Połącz dane godzinowe z innymi danymi
for df in all_new_data:
    combined_df = combined_df.merge(df, on="timestamp", how="outer")

# Usuń dane poza zakresem min_timestamp i max_timestamp
combined_df = combined_df[(combined_df["timestamp"] >= min_timestamp) & (combined_df["timestamp"] <= max_timestamp)]

# Przekształć wartości co2_price na PLN, mnożąc przez pln_eur
combined_df["co2_price"] = combined_df["co2_price"] * combined_df["pln_eur"]

# Przekształć wartości Brent_USD na PLN, mnożąc przez pln_usd, i zmień nazwę kolumny na brent_price
combined_df["brent_price"] = combined_df["Brent_USD"] * combined_df["pln_usd"]
combined_df = combined_df.drop(columns=["Brent_USD"])

# Przekształć ceny rynków zagranicznych (sk_price, se_price, cz_price, lt_price) na PLN, mnożąc przez pln_eur
foreign_price_cols = ["sk_price", "se_price", "cz_price", "lt_price"]
for col in foreign_price_cols:
    if col in combined_df.columns:
        combined_df[col] = combined_df[col] * combined_df["pln_eur"]

# Dodaj kolumnę dzień tygodnia
combined_df["day_of_week"] = combined_df["timestamp"].dt.dayofweek  # 0 = poniedziałek, 6 = niedziela

# Dodanie zmiennej month (1-12)
combined_df["month"] = combined_df["timestamp"].dt.month

# Dodanie zmiennej hour (0-23)
combined_df["hour"] = combined_df["timestamp"].dt.hour

# Średnie kroczące dla cen energii
combined_df['fixing_i_price_mean24'] = combined_df["fixing_i_price"].shift(1).rolling(window=24, min_periods=1).mean()
combined_df['fixing_i_price_mean48'] = combined_df["fixing_i_price"].shift(1).rolling(window=48, min_periods=1).mean()

# Obliczanie lagów
lags = [24, 48, 72, 96, 120, 144, 168]
for lag in lags:
    col_name = f"fixing_i_price_lag{lag}"
    combined_df[col_name] = combined_df["fixing_i_price"].shift(lag)
    # Wypełnij brakujące wartości na początku wartościami z fixing_i_price
    combined_df.loc[:lag-1, col_name] = combined_df.loc[:lag-1, "fixing_i_price"]

# Sprawdzenie wyniku
print("Przykładowe dane po dodaniu nowych zmiennych:")
print(combined_df[["timestamp", "month", "fixing_i_price", "fixing_i_price_lag24", "fixing_i_price_lag168"]].head())

# Dodaj kolumnę is_holiday (używając biblioteki holidays dla Polski)
holidays = hl.PL(years=range(2016, 2025)).keys()
print(f"\nLista dni wolnych w Polsce: {holidays}")
combined_df["is_holiday"] = combined_df["timestamp"].dt.date.isin(holidays)

# Zamień wartości w kolumnie is_holiday na 1 (True) lub 0 (False)
combined_df["is_holiday"] = combined_df["is_holiday"].astype(int)

# Usuń kolumnę gas_volume
if "gas_volume" in combined_df.columns:
    combined_df = combined_df.drop(columns=["gas_volume"])

combined_df["peak_hour"] = 0
# Ustaw wartość peak_hour w zależności od warunków
combined_df["peak_hour"] = combined_df.apply(
    lambda row: 1 if (
        (row["is_holiday"] == 1 or row["day_of_week"] in [5, 6]) and row["Load"] > 18000 and row["hour"] in [7,8,9,16,17,18]
    ) or (
        (row["is_holiday"] == 0 and row["day_of_week"] not in [5, 6]) and row["Load"] > 23000 and row["hour"] in [7,8,9,16,17,18]
    ) else 0,
    axis=1
)

# Wyświetl liczbę wartości 1 w kolumnie peak_hour
peak_hour_count = combined_df["peak_hour"].sum()
high_load_count = combined_df[combined_df["Load"] > 23000].shape[0]
print(f"Liczba wartości 1 w kolumnie peak_hour: {peak_hour_count}")
print(f"Liczba rekordów z Load > 23000: {high_load_count}")

# Posortuj według timestamp
combined_df = combined_df.sort_values("timestamp")

# Usuń rekordy, gdzie timestamp ma niezerowe minuty
combined_df = combined_df[combined_df['timestamp'].dt.minute == 0]

# Wyświetl całkowitą liczbę brakujących wartości (NaN) przed obróbką
total_missing_values = combined_df.isna().sum().sum()
print(f"\nCałkowita liczba brakujących wartości (NaN) przed obróbką: {total_missing_values}")

# Deduplikacja timestampów (zachowaj pierwszy rekord)
print(f"\nLiczba rekordów przed deduplikacją: {len(combined_df)}")
duplicates_before = combined_df['timestamp'].duplicated().sum()
print(f"Liczba zduplikowanych timestampów przed deduplikacją: {duplicates_before}")
combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='first')
print(f"Liczba rekordów po deduplikacji: {len(combined_df)}")

# Sprawdź duplikaty timestampów po deduplikacji
duplicates_after = combined_df['timestamp'].duplicated().sum()
print(f"Liczba zduplikowanych timestampów po deduplikacji: {duplicates_after}")

# Weryfikacja liczby rekordów
print(f"\nMinimalny timestamp: {combined_df['timestamp'].min()}")
print(f"Maksymalny timestamp: {combined_df['timestamp'].max()}")

# Oblicz oczekiwaną liczbę rekordów
expected_records = (combined_df['timestamp'].max() - combined_df['timestamp'].min()).total_seconds() / 3600 + 1
print(f"Oczekiwana liczba rekordów: {int(expected_records)}")
print(f"Rzeczywista liczba rekordów: {len(combined_df)}")

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

# Wyświetl informacje o brakujących wartościach i zapisz statystyki do pliku tekstowego PRZED interpolacją
print("\nLiczba brakujących wartości (NaN) w każdej kolumnie przed interpolacją:")
missing_values = combined_df.isna().sum()
print(missing_values)

# Przygotuj statystyki tekstowe o brakujących wartościach
missing_stats_file = os.path.join(output_dir, "missing_values_stats_2016_2023.txt")
with open(missing_stats_file, "w", encoding="utf-8") as f:
    f.write("Statystyki brakujących wartości (NaN) w danych przed interpolacją:\n\n")
    total_records = len(combined_df)
    for column in combined_df.columns:
        missing_count = combined_df[column].isna().sum()
        if missing_count > 0:
            missing_percentage = (missing_count / total_records) * 100
            f.write(f"Kolumna '{column}': {missing_count} brakujących wartości ({missing_percentage:.2f}% z {total_records} rekordów)\n")
            # Wylistuj timestampy z brakującymi wartościami
            missing_timestamps = combined_df[combined_df[column].isna()]["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
            f.write("Brakujące wartości w następujących timestampach:\n")
            f.write("\n".join(missing_timestamps[:1000]))  # Ograniczamy do 1000 timestampów, aby plik nie był zbyt duży
            if len(missing_timestamps) > 1000:
                f.write(f"\n... i {len(missing_timestamps) - 1000} więcej timestampów ...\n")
            f.write("\n\n")
        else:
            f.write(f"Kolumna '{column}': Brak brakujących wartości.\n\n")

print(f"\nStatystyki brakujących wartości zapisano do pliku: {missing_stats_file}")

# Interpolacja liniowa dla wszystkich kolumn numerycznych
numeric_cols = combined_df.select_dtypes(include=['float64', 'int64']).columns

# Liczniki dla ffill i bfill
ffill_counts = {}
bfill_counts = {}

# Krok 1: Interpolacja liniowa
combined_df_before_ffill = combined_df[numeric_cols].copy()
combined_df[numeric_cols] = combined_df[numeric_cols].interpolate(method='linear')

# Krok 2: ffill - sprawdź, ile wartości zostało wypełnionych
combined_df_after_ffill = combined_df[numeric_cols].copy()
combined_df[numeric_cols] = combined_df[numeric_cols].ffill()
for col in numeric_cols:
    ffill_count = (combined_df_after_ffill[col].isna() & ~combined_df[numeric_cols][col].isna()).sum()
    if ffill_count > 0:
        ffill_counts[col] = ffill_count

# Krok 3: bfill - sprawdź, ile wartości zostało wypełnionych
combined_df_after_bfill = combined_df[numeric_cols].copy()
combined_df[numeric_cols] = combined_df[numeric_cols].bfill()
for col in numeric_cols:
    bfill_count = (combined_df_after_bfill[col].isna() & ~combined_df[numeric_cols][col].isna()).sum()
    if bfill_count > 0:
        bfill_counts[col] = bfill_count

# Wyświetl informacje o ffill i bfill
print("\nWyniki użycia ffill():")
if ffill_counts:
    for col, count in ffill_counts.items():
        print(f"Kolumna '{col}': Wypełniono {count} wartości metodą ffill.")
else:
    print("Metoda ffill() nie była potrzebna - wszystkie wartości zostały wypełnione przez interpolację liniową.")

print("\nWyniki użycia bfill():")
if bfill_counts:
    for col, count in bfill_counts.items():
        print(f"Kolumna '{col}': Wypełniono {count} wartości metodą bfill.")
else:
    print("Metoda bfill() nie była potrzebna - wszystkie wartości zostały wypełnione przez interpolację liniową lub ffill().")

# Sprawdź, czy po interpolacji nadal są brakujące wartości
print("\nLiczba brakujących wartości (NaN) w każdej kolumnie po interpolacji:")
print(combined_df.isna().sum())

# Zapisz do nowego pliku CSV
combined_df.to_csv(output_file, index=False, encoding="utf-8")
print(f"\nDane połączone zapisane do pliku: {output_file}")
print(f"Liczba rekordów: {len(combined_df)}")
print(f"Kolumny w pliku: {combined_df.columns.tolist()}")

# Wyświetl pierwsze 5 wierszy
print("\nPierwsze 5 wierszy danych:")
print(combined_df.head())
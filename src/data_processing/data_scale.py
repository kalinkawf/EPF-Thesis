import pandas as pd
import os
from datetime import datetime, timedelta
import holidays as hl

# Ścieżki do plików
input_dir = "../../data/processed_data/"
other_data_input_dir = "../../data/other_data/"
output_file = "../../data/database.csv"

# Lista plików i ich kolumn z czasem
files = {
    "weather_hourly.csv": "Time",
    "power_outages.csv": "Time",
    "import_export.csv": "Time",
    "energy_sources_prod.csv": "date",
    "electricity_prices_day_ahead_hourly_all.csv": "date",
    "load.csv": "date"
}

price_files = {
    "prices_gas_day_ahead_all.csv": ["gas_price", "gas_volume"],
    "prices_coal_all.csv": ["coal_pscmi1_pln_per_gj"],
    "prices_eu_co2.csv": ["co2_price"],
    "usd_eur_pln_daily.csv": ["pln_usd", "pln_eur"],
    "Europe_Brent_Spot_Price.csv": ["Brent_USD"],
}

# 78843 - tyle danych przed dodaniem load
# 78813 - tyle danych po dodaniu load

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

# Przekształć wartości co2_price na PLN, mnożąc przez pln_eur
combined_df["co2_price"] = combined_df["co2_price"] * combined_df["pln_eur"]
combined_df = combined_df.drop(columns=["pln_eur"])

# Przekształć wartości Brent_USD na PLN, mnożąc przez pln_usd, i zmień nazwę kolumny na brent_price
combined_df["brent_price"] = combined_df["Brent_USD"] * combined_df["pln_usd"]
combined_df = combined_df.drop(columns=["Brent_USD"])

# Dodaj kolumnę dzień tygodnia
combined_df["day_of_week"] = combined_df["timestamp"].dt.dayofweek  # 0 = poniedziałek, 6 = niedziela

# Dodanie zmiennej month (1-12)
combined_df["month"] = combined_df["timestamp"].dt.month

# Dodanie zmiennej hour (0-23)
combined_df["hour"] = combined_df["timestamp"].dt.hour

combined_df["fixing_i_price_lag24"] = combined_df["fixing_i_price"].shift(24)  # Cena w tej samej godzinie poprzedniego dnia
combined_df["fixing_i_price_lag168"] = combined_df["fixing_i_price"].shift(168)  # Cena w tej samej godzinie poprzedniego tygodnia

# Wypełnienie brakujących wartości w lag24 wartościami z fixing_i_price dla pierwszych 24 rekordów
combined_df.loc[:23, "fixing_i_price_lag24"] = combined_df.loc[:23, "fixing_i_price"]

# Wypełnienie brakujących wartości w lag168 wartościami z fixing_i_price dla pierwszych 168 rekordów
combined_df.loc[:167, "fixing_i_price_lag168"] = combined_df.loc[:167, "fixing_i_price"]

# Sprawdzenie wyniku
print("Przykładowe dane po dodaniu nowych zmiennych:")
print(combined_df[["timestamp", "month", "fixing_i_price", "fixing_i_price_lag24", "fixing_i_price_lag168"]].head())

# Dodaj kolumnę is_holiday (używając biblioteki holidays dla Polski)
holidays = hl.PL(years=range(2016, 2025)).keys()
print(f"\nLista dni wolnych w Polsce: {holidays}")
combined_df["is_holiday"] = combined_df["timestamp"].dt.date.isin(holidays)

# Zamień wartości w kolumnie is_holiday na 1 (True) lub 0 (False)
combined_df["is_holiday"] = combined_df["is_holiday"].astype(int)

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
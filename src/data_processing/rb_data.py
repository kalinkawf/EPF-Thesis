import pandas as pd
import os
from datetime import datetime

# Ścieżki
input_path = "../../data/rb_data"
output_path = "../../data/processed_data"
os.makedirs(output_path, exist_ok=True)

# Lista lat i miesięcy
years = range(2016, 2024)
months = range(1, 13)

# Inicjalizacja DataFrame do przechowywania danych
rb_data = pd.DataFrame()

# Wczytywanie danych dla każdego roku i miesiąca
for year in years:
    for month in months:
        # Wzorzec nazwy pliku: PL_CENY_ROZL_RB_YYYYMMDD_YYYYMMDD_*.csv
        # Szukamy pliku pasującego do roku i miesiąca
        for file_name in os.listdir(input_path):
            if file_name.startswith(f"PL_CENY_ROZL_RB_{year}{month:02d}01"):
                file_path = os.path.join(input_path, file_name)
                # Wczytanie pliku CSV
                df = pd.read_csv(file_path, sep=";")
                
                # Tworzenie kolumny Timestamp na podstawie Data i Godzina
                # Data w formacie YYYYMMDD, Godzina jako liczba (1-24), zamieniamy na format datetime
                df["Timestamp"] = df.apply(
                    lambda row: datetime.strptime(f"{row['Data']} {int(row['Godzina']) - 1:02d}:00:00", "%Y%m%d %H:%M:%S"),
                    axis=1
                )
                
                # Wybór kolumny CRO i zmiana nazwy na RB_price
                # Zamiana przecinków na kropki i konwersja na float
                df["RB_price"] = df["CRO"].str.replace(",", ".").replace("-", float("nan")).astype(float)
                
                # Wybór tylko potrzebnych kolumn
                df = df[["Timestamp", "RB_price"]]
                
                # Dołączanie do głównego DataFrame
                rb_data = pd.concat([rb_data, df], ignore_index=True)

# Sortowanie po Timestamp
rb_data.sort_values("Timestamp", inplace=True)

# Sprawdzenie brakujących danych (NaN)
print("\nStatystyki brakujących danych (NaN):")
missing_count = rb_data["RB_price"].isna().sum()
total_count = len(rb_data)
missing_percentage = (missing_count / total_count) * 100
print(f"RB_price: {missing_count} brakujących wartości ({missing_percentage:.2f}% z {total_count} rekordów)")

# Wyświetlenie i zapis brakujących wartości
if missing_count > 0:
    missing_data = rb_data[rb_data["RB_price"].isna()][["Timestamp"]]
    print("\nBrakujące wartości dla RB_price:")
    print(missing_data)
    # Zapis do pliku CSV
    # missing_file = os.path.join(output_path, "missing_rb_price_2016_2023.csv")
    # missing_data.to_csv(missing_file, index=False)
    # print(f"\nZapisano brakujące wartości do pliku: {missing_file}")

# Zapis do pliku CSV
output_file = os.path.join(output_path, "rb_data.csv")
rb_data.to_csv(output_file, index=False)
print(f"\nZapisano plik: {output_file}")
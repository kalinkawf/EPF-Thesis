import pandas as pd
import os
from datetime import datetime

# Ścieżki
input_path = "../../data/foreign_markets"
output_path = "../../data/processed_data"
os.makedirs(output_path, exist_ok=True)

# Lista rynków i lat
markets = ["CZ", "SE", "LT", "SK"]
years = range(2016, 2024)

# Inicjalizacja słownika do przechowywania danych dla każdego rynku
market_data = {market: pd.DataFrame() for market in markets}

# Wczytywanie danych dla każdego rynku i roku
for market in markets:
    for year in years:
        file_name = f"{market}_{year}.csv"
        file_path = os.path.join(input_path, file_name)
        if os.path.exists(file_path):
            # Wczytanie pliku CSV
            df = pd.read_csv(file_path)
            
            # Parsowanie kolumny MTU (CET/CEST) na Timestamp
            # Usuwamy sufiks (CET) lub (CEST) i bierzemy początek przedziału czasowego
            df["Timestamp"] = df["MTU (CET/CEST)"].apply(
                lambda x: datetime.strptime(x.split(" - ")[0].replace(" (CET)", "").replace(" (CEST)", ""), "%d/%m/%Y %H:%M:%S")
            )
            
            # Wybór kolumny z ceną Day-ahead
            df = df[["Timestamp", "Day-ahead Price (EUR/MWh)"]].rename(columns={"Day-ahead Price (EUR/MWh)": f"{market.lower()}_price"})
            
            # Dołączanie do danych dla danego rynku
            market_data[market] = pd.concat([market_data[market], df], ignore_index=True)

# Połączenie danych w jeden DataFrame
# Najpierw ustawiamy Timestamp jako indeks dla każdego rynku
for market in markets:
    market_data[market].set_index("Timestamp", inplace=True)

# Łączenie danych po Timestamp
merged_df = market_data["CZ"].join([market_data["SE"], market_data["LT"], market_data["SK"]], how="outer")

# Reset indeksu, aby Timestamp stał się kolumną
merged_df.reset_index(inplace=True)

# Sortowanie po Timestamp
merged_df.sort_values("Timestamp", inplace=True)

# Sprawdzenie brakujących danych (NaN)
print("\nStatystyki brakujących danych (NaN):")
missing_data = {}
for column in ["cz_price", "se_price", "lt_price", "sk_price"]:
    missing_count = merged_df[column].isna().sum()
    total_count = len(merged_df)
    missing_percentage = (missing_count / total_count) * 100
    print(f"{column}: {missing_count} brakujących wartości ({missing_percentage:.2f}% z {total_count} rekordów)")
    # Zapis miejsc z brakującymi wartościami
    if missing_count > 0:
        missing_data[column] = merged_df[merged_df[column].isna()][["Timestamp"]]

# Wyświetlenie i zapis brakujących wartości dla sk_price
if "sk_price" in missing_data:
    print("\nBrakujące wartości dla sk_price:")
    print(missing_data["sk_price"])
    # Zapis do pliku CSV
    # missing_sk_file = os.path.join(output_path, "missing_sk_price_2016_2023.csv")
    # missing_data["sk_price"].to_csv(missing_sk_file, index=False)
    # print(f"\nZapisano brakujące wartości dla sk_price do pliku: {missing_sk_file}")

# Zapis głównego pliku CSV
output_file = os.path.join(output_path, "foreign_prices.csv")
merged_df.to_csv(output_file, index=False)
print(f"\nZapisano plik: {output_file}")
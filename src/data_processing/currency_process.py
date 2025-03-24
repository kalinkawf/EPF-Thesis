import pandas as pd
import os

# Ścieżka do folderu z plikami
input_folder = "../../data/kursy/"
output_file = "../../data/other_data/usd_eur_pln_daily.csv"

def process_currency_files(input_folder, output_file):
    # Lista do przechowywania przetworzonych danych
    all_data = []

    # Iteracja przez wszystkie pliki w folderze
    for year in range(2016, 2025):  # Zakres lat od 2016 do 2024
        file_name = f"archiwum_tab_a_{year}.csv"
        file_path = os.path.join(input_folder, file_name)
        print(f"Przetwarzanie pliku: {file_name}")
        
        try:
            # Wczytaj plik CSV
            df = pd.read_csv(file_path, delimiter=";", encoding="utf-8")
            
            # Sprawdź, czy wymagane kolumny istnieją
            if "data" not in df.columns or "1USD" not in df.columns or "1EUR" not in df.columns:
                print(f"Błąd: Brak wymaganych kolumn w pliku {file_name}")
                continue
            
            # Wybierz tylko kolumny 'data', '1USD' i '1EUR'
            df = df[["data", "1USD", "1EUR"]]
            
            # Konwertuj kolumnę 'data' na datetime i nazwij ją 'timestamp'
            df["timestamp"] = pd.to_datetime(df["data"], format="%Y%m%d", errors="coerce")
            
            # Usuń wiersze z błędnymi datami
            df = df.dropna(subset=["timestamp"])
            
            # Zamień przecinki na kropki w kolumnach '1USD' i '1EUR' i przekonwertuj na float
            df["1USD"] = df["1USD"].str.replace(",", ".").astype(float)
            df["1EUR"] = df["1EUR"].str.replace(",", ".").astype(float)
            
            # Dodaj przetworzone dane do listy
            all_data.append(df[["timestamp", "1USD", "1EUR"]])
        except Exception as e:
            print(f"Błąd podczas przetwarzania pliku {file_name}: {e}")
            continue

    # Połącz wszystkie dane w jeden DataFrame
    try:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Posortuj dane po kolumnie 'timestamp'
        combined_df = combined_df.sort_values(by="timestamp")
        
        # Zmień nazwy kolumn na 'plnusd' i 'pln_eur'
        combined_df = combined_df.rename(columns={"1USD": "plnusd", "1EUR": "pln_eur"})

        # Zapisz dane do pliku CSV
        combined_df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"Przetwarzanie zakończone. Wynik zapisano w pliku: {output_file}")
    except Exception as e:
        print(f"Błąd podczas łączenia danych: {e}")

# Uruchomienie funkcji
process_currency_files(input_folder, output_file)
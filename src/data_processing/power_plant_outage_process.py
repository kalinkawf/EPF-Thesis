import pandas as pd
import os


# Ścieżka do folderu z plikami CSV
input_folder = "../../data/power_outage/"
# Ścieżka do pliku wyjściowego
output_file = "../../data/power_outages.csv"

def process_csv_files(input_folder, output_file):
    # Lista do przechowywania przetworzonych danych
    all_data = []

    # Iteracja przez wszystkie pliki CSV w folderze
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_folder, file_name)
            print(f"Przetwarzanie pliku: {file_name}")
            
            # Wczytanie pliku CSV
            df = pd.read_csv(file_path)
            
            # Upewnij się, że kolumna 'index' jest traktowana jako czas
            df['index'] = pd.to_datetime(df['index'])
            
            # Grupowanie danych po czasie i sumowanie strat
            grouped = df.groupby('index').agg({
                'Power plant loss [MW]': 'sum',
                'Network loss [MW]': 'sum'
            }).reset_index()
            
            # Zmiana nazw kolumn na wymagane
            grouped.rename(columns={
                'index': 'Time',
                'Power plant loss [MW]': 'power_loss',
                'Network loss [MW]': 'Network_loss'
            }, inplace=True)
            
            # Dodanie przetworzonych danych do listy
            all_data.append(grouped)

    # Połączenie wszystkich przetworzonych danych w jeden DataFrame
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Sortowanie danych po czasie
    final_df.sort_values(by='Time', inplace=True)
    
    # Zapisanie do pliku CSV
    final_df.to_csv(output_file, index=False)
    print(f"Przetwarzanie zakończone. Wynik zapisano w pliku: {output_file}")

# Uruchomienie funkcji
process_csv_files(input_folder, output_file)
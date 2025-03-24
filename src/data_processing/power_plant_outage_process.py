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
            if "mocy" in file_name:
                df = pd.read_csv(file_path, delimiter=";")
            else:
                df = pd.read_csv(file_path, delimiter=",")
            
            # Sprawdzenie struktury pliku na podstawie nagłówków
            if "index" in df.columns:  # Stary format
                df['index'] = pd.to_datetime(df['index'])
                grouped = df.groupby('index').agg({
                    'Power plant loss [MW]': 'sum',
                    'Network loss [MW]': 'sum'
                }).reset_index()
                grouped.rename(columns={
                    'index': 'Time',
                    'Power plant loss [MW]': 'power_loss',
                    'Network loss [MW]': 'Network_loss'
                }, inplace=True)
                all_data.append(grouped)
            elif "Doba" in df.columns:  # Nowy format
                # Tworzenie kolumny 'Time' na podstawie 'Doba' i 'OREB [Jednostka czasu od-do]'
                df['Time'] = pd.to_datetime(
                    df['Doba'] + " " + df['OREB [Jednostka czasu od-do]'].str.split(" - ").str[0],
                    format="%Y-%m-%d %H:%M"
                )
                grouped = df.groupby('Time').agg({
                    'Ubytki elektrowniane [MW]': 'sum',
                    'Ubytki sieciowe [MW]': 'sum'
                }).reset_index()
                grouped.rename(columns={
                    'Ubytki elektrowniane [MW]': 'power_loss',
                    'Ubytki sieciowe [MW]': 'Network_loss'
                }, inplace=True)
                
                # Konwersja na godzinowe interwały i podział przez 4
                grouped.set_index('Time', inplace=True)
                hourly_data = grouped.resample('h').sum() / 4  # Sumowanie i dzielenie przez 4
                hourly_data.reset_index(inplace=True)
                all_data.append(hourly_data)
            else:
                print(f"Nieznany format pliku: {file_name}")
                continue

    # Połączenie wszystkich przetworzonych danych w jeden DataFrame
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Sortowanie danych po czasie
    final_df.sort_values(by='Time', inplace=True)
    
    # Zapisanie do pliku CSV
    final_df.to_csv(output_file, index=False)
    print(f"Przetwarzanie zakończone. Wynik zapisano w pliku: {output_file}")

# Uruchomienie funkcji
process_csv_files(input_folder, output_file)
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
                    format="%Y-%m-%d %H:%M",
                    errors='coerce'  # Zamienia błędne wartości na NaT
                )

                # Sprawdzenie, czy są jakieś błędne wartości w 'Time'
                if df['Time'].isna().any():
                    print("Uwaga: Niektóre wartości w kolumnie 'Time' nie mogły być przekonwertowane na datetime:")
                    print(df[df['Time'].isna()][['Doba', 'OREB [Jednostka czasu od-do]']])

                # Usunięcie rekordów z błędnymi wartościami 'Time' (opcjonalne)
                df = df.dropna(subset=['Time'])

                # Grupowanie po 'Time' i sumowanie wartości dla wszystkich jednostek w danym interwale
                grouped = df.groupby('Time').agg({
                    'Ubytki elektrowniane [MW]': 'sum',
                    'Ubytki sieciowe [MW]': 'sum'
                }).reset_index()

                # Zmiana nazw kolumn
                grouped.rename(columns={
                    'Ubytki elektrowniane [MW]': 'power_loss',
                    'Ubytki sieciowe [MW]': 'Network_loss'
                }, inplace=True)

                # Ustawienie 'Time' jako indeks
                grouped.set_index('Time', inplace=True)

                # Resampling do interwałów godzinowych
                # Najpierw upewniamy się, że indeks jest w formacie datetime
                grouped.index = pd.to_datetime(grouped.index)

                print(grouped.head())

                # Resampling z użyciem średniej
                # Używamy 'H' (godzinowe interwały), a funkcja mean() automatycznie oblicza średnią z dostępnych rekordów
                hourly_data = grouped.resample('H').mean()

                print(hourly_data.head())

                # Podzielenie wartości Network_loss przez 4
                hourly_data['Network_loss'] = hourly_data['Network_loss'] / 4

                # Wypełnienie brakujących wartości zerami (jeśli w danej godzinie nie ma żadnych danych)
                hourly_data.fillna(0, inplace=True)

                # Reset indeksu
                hourly_data.reset_index(inplace=True)

                # Sprawdzenie wyniku
                print("Przykładowe dane po resamplingu:")
                print(hourly_data.head())

                # Dodanie do listy all_data
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
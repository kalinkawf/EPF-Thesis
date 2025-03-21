import pandas as pd
import glob

# Ścieżka do folderu z plikami CSV
folder_path = "../../data/import_export"
pse_output_file = "../../data/import_export/import_export_pse.csv"
instrat_input_file = "../../data/import_export/PSE_electricity_import_export.csv"
output_file = "../../data/import_export.csv"

# Pobierz wszystkie pliki CSV z folderu
csv_files = glob.glob(folder_path + "/Import_export_2024*.csv")

# Lista do przechowywania przetworzonych danych
all_data = []

for file in csv_files:
    print(f"Przetwarzam plik: {file}")
    
    # Wczytaj plik CSV
    df = pd.read_csv(file, delimiter=";")
    
    # Ustaw poprawne nazwy kolumn
    df.columns = [
        'Doba handlowa', 'OREB', 'Czechy Eksport [MW]', 'Czechy Import [MW]',
        'Słowacja Eksport [MW]', 'Słowacja Import [MW]', 'Niemcy Eksport [MW]', 'Niemcy Import [MW]',
        'Szwecja Eksport [MW]', 'Szwecja Import [MW]', 'Ukraina Eksport [MW]', 'Ukraina Import [MW]',
        'Litwa Eksport [MW]', 'Litwa Import [MW]', 'Data publikacji'
    ]
    
    # Napraw błędne formaty czasu w kolumnie 'OREB'
    df['OREB'] = df['OREB'].str.replace("02a", "02", regex=False)

    # Tworzenie kolumny 'Time', obsługa błędnych czasów
    df['Time'] = pd.to_datetime(
        df['Doba handlowa'] + " " + df['OREB'].str.split(" - ").str[0],
        format="%Y-%m-%d %H:%M",
        errors='coerce'
    )

    # Usuń wiersze z nieprawidłowym czasem
    df = df.dropna(subset=['Time'])

    # Obliczanie bilansu dla każdego kraju
    df['Niemcy Bilans'] = (df['Niemcy Eksport [MW]'] + df['Niemcy Import [MW]']) * 0.25
    df['Czechy Bilans'] = (df['Czechy Eksport [MW]'] + df['Czechy Import [MW]']) * 0.25
    df['Litwa Bilans'] = (df['Litwa Eksport [MW]'] + df['Litwa Import [MW]']) * 0.25
    df['Słowacja Bilans'] = (df['Słowacja Eksport [MW]'] + df['Słowacja Import [MW]']) * 0.25
    df['Szwecja Bilans'] = (df['Szwecja Eksport [MW]'] + df['Szwecja Import [MW]']) * 0.25
    df['Ukraina Bilans'] = (df['Ukraina Eksport [MW]'] + df['Ukraina Import [MW]']) * 0.25
    
    # Wybór interesujących kolumn
    df = df[['Time', 'Niemcy Bilans', 'Czechy Bilans', 'Litwa Bilans', 'Słowacja Bilans', 'Szwecja Bilans', 'Ukraina Bilans']]
    
    # Grupowanie danych do granulacji godzinowej
    df_hourly = df.resample('h', on='Time').sum()
    
    # Resetowanie indeksu po grupowaniu
    df_hourly.reset_index(inplace=True)
    
    # Dodanie przetworzonych danych do listy
    all_data.append(df_hourly)

# Połączenie wszystkich przetworzonych danych w jeden DataFrame
final_df = pd.concat(all_data)

# Zapisanie do nowego pliku CSV
final_df.to_csv(pse_output_file, index=False, encoding="utf-8")

print(f"Przetworzono pliki i zapisano wynik do {pse_output_file}")

instat_df = pd.read_csv(instrat_input_file, delimiter=",")
# Zmiana nazw kolumn w instat_df
instat_df.columns = [
    'Time', 'Niemcy Bilans', 'Czechy Bilans', 'Litwa Bilans',
    'Słowacja Bilans', 'Szwecja Bilans', 'Ukraina Bilans'
]

# Połącz dane z obu plików
combined_df = pd.concat([instat_df, final_df])

# Posortuj dane według kolumny 'Time'
combined_df['Time'] = pd.to_datetime(combined_df['Time'])
combined_df = combined_df.sort_values(by='Time')

combined_df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Przetworzono pliki i zapisano wynik do {output_file}")
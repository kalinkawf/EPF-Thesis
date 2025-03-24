import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path_to_data = "../../data"
files = {
    "weather_dataset.csv": f"{path_to_data}/weather_dataset.csv",
    "power_outages.csv": f"{path_to_data}/power_outages.csv",
    "import_export.csv": f"{path_to_data}/import_export.csv",
    "energy_sources_prod.csv": f"{path_to_data}/energy_sources_prod.csv",
    "electricity_prices_day_ahead_hourly_all.csv": f"{path_to_data}/electricity_prices_day_ahead_hourly_all.csv"
}

# Plik wyjściowy dla statystyk
output_stats_file = "../../data/statistics_summary_extended.txt"
output_plots_dir = "../../data/plots/"

# Utwórz katalog na wykresy, jeśli nie istnieje
import os
if not os.path.exists(output_plots_dir):
    os.makedirs(output_plots_dir)

# Funkcja do obliczania statystyk dla DataFrame
def calculate_statistics(df, file_name, date_column=None):
    stats = []
    stats.append(f"\n=== Statystyki dla pliku: {file_name} ===")
    stats.append(f"Liczba rekordów: {len(df)}")
    stats.append(f"Kolumny: {df.columns.tolist()}")
    
    # Zakres czasowy, jeśli istnieje kolumna z datą
    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        min_date = df[date_column].min()
        max_date = df[date_column].max()
        stats.append(f"Zakres czasowy: od {min_date} do {max_date}")
    
    # Statystyki dla kolumn numerycznych
    stats.append("\nStatystyki dla kolumn numerycznych:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        stats.append(f"\nKolumna: {col}")
        stats.append(f"Średnia: {df[col].mean():.2f}")
        stats.append(f"Mediana: {df[col].median():.2f}")
        stats.append(f"Odchylenie standardowe: {df[col].std():.2f}")
        stats.append(f"Percentyl 25%: {df[col].quantile(0.25):.2f}")
        stats.append(f"Percentyl 75%: {df[col].quantile(0.75):.2f}")
        stats.append(f"Min: {df[col].min():.2f}")
        stats.append(f"Max: {df[col].max():.2f}")
        stats.append(f"Liczba brakujących wartości (NaN): {df[col].isna().sum()}")
        stats.append(f"Liczba wartości zerowych: {(df[col] == 0).sum()}")
    
    # Typy danych
    stats.append("\nTypy danych w kolumnach:")
    for col in df.columns:
        stats.append(f"{col}: {df[col].dtype}")
    
    # Pierwsze 5 wierszy
    stats.append("\nPierwsze 5 wierszy danych:")
    stats.append(df.head().to_string())
    
    return stats

# Funkcja do generowania wykresów
def generate_plots(df, file_name, date_column=None):
    # Wybierz kolumny numeryczne
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Szereg czasowy (jeśli istnieje kolumna z datą)
    if date_column and date_column in df.columns:
        plt.figure(figsize=(12, 6))
        for col in numeric_cols:
            plt.plot(df[date_column], df[col], label=col)
        plt.title(f"Szereg czasowy - {file_name}")
        plt.xlabel("Czas")
        plt.ylabel("Wartość")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_plots_dir}/{file_name}_timeseries.png")
        plt.close()
    
    # Histogramy
    for col in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col].dropna(), bins=50, kde=True)
        plt.title(f"Histogram - {col} ({file_name})")
        plt.xlabel(col)
        plt.ylabel("Liczba wystąpień")
        plt.savefig(f"{output_plots_dir}/{file_name}_{col}_histogram.png")
        plt.close()

# Lista do przechowywania wszystkich statystyk
all_stats = []

# Analiza każdego pliku
for file_name, file_path in files.items():
    try:
        print(f"Wczytuję plik: {file_name}")
        df = pd.read_csv(file_path)
        
        # Określ kolumnę z datą dla każdego pliku
        date_column = None
        if file_name == "weather_dataset.csv":
            date_column = "timestamp"
        elif file_name == "power_outages.csv":
            date_column = "Time"
        elif file_name == "import_export.csv":
            date_column = "Time"
        elif file_name == "energy_sources_prod.csv":
            date_column = "date"
        elif file_name == "electricity_prices_day_ahead_hourly_all.csv":
            date_column = "date"
        
        # Oblicz statystyki
        file_stats = calculate_statistics(df, file_name, date_column)
        all_stats.extend(file_stats)
        
        # Generuj wykresy
        generate_plots(df, file_name, date_column)
    
    except Exception as e:
        all_stats.append(f"\nBłąd podczas wczytywania pliku {file_name}: {str(e)}")

# Wyświetl statystyki w konsoli
for line in all_stats:
    print(line)

# Zapisz statystyki do pliku tekstowego
with open(output_stats_file, "w", encoding="utf-8") as f:
    for line in all_stats:
        f.write(line + "\n")

print(f"\nStatystyki zapisane do pliku: {output_stats_file}")
print(f"Wykresy zapisane w katalogu: {output_plots_dir}")
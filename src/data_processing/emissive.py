import pandas as pd

# Wczytaj dane
file_path = "../../data/other_data/emissive_vs_non_emissive_all.csv" 
df = pd.read_csv(file_path)

# Konwersja kolumny 'date' na datetime
df["Time"] = pd.to_datetime(df["date"], format="%d.%m.%Y %H:%M")

# Usuń niepotrzebne kolumny
df = df.drop(columns=["date", "date_utc", "emissive_sources", "emissive_sources_percentage", "non_emissive_sources"])

# Ustawienie indeksu na 'timestamp' i wykonanie resamplingu do danych godzinowych
df = df.set_index("Time").resample("h").mean().reset_index()

print(df.head())

output_file = "../../data/processed_data/non_emissive.csv"
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"Dane zapisane do pliku: {output_file}")
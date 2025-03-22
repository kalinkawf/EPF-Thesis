import pandas as pd

# Ścieżka do pliku
file_path = "../data/weather_dataset.csv"

# Wczytaj dane
print(f"Wczytuję plik: {file_path}")
df = pd.read_csv(file_path)

# 1. Sprawdź podstawowe informacje o danych
print("\nInformacje o danych:")
print(df.info())

# 2. Sprawdź liczbę rekordów
expected_records = (9 * 365 + 2) * 3  # 9 lat (2016-2024), 365 dni + 2 dni przestępne, 3 pomiary dziennie (6:00, 12:00, 18:00)
print(f"\nOczekiwana liczba rekordów: {expected_records}")
print(f"Rzeczywista liczba rekordów: {len(df)}")

# 3. Sprawdź braki danych (NaN)
print("\nLiczba brakujących wartości (NaN) w każdej kolumnie:")
print(df.isna().sum())

# 4. Statystyki opisowe dla kolumn numerycznych
numeric_columns = [col for col in df.columns if col != "timestamp"]
print("\nStatystyki opisowe dla kolumn numerycznych:")
print(df[numeric_columns].describe())

# 5. Sprawdź unikalne godziny w timestamp
df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
print("\nUnikalne godziny w timestamp:")
print(df["hour"].unique())

# 6. Wyświetl pierwsze 5 wierszy
print("\nPierwsze 5 wierszy danych:")
print(df.head())

# 7. Wyświetl ostatnie 5 wierszy
print("\nOstatnie 5 wierszy danych:")
print(df.tail())

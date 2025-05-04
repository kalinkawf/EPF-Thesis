import pandas as pd
import numpy as np

# Wczytanie danych
df = pd.read_csv("..\..\data\database.csv")
df2 = pd.read_csv("..\..\data\short_database.csv")  # Upewnij się, że ścieżki są poprawne

# Sprawdzenie brakujących wartości w pełnym zbiorze danych (df)
print("### Analiza brakujących wartości w pełnym zbiorze danych (database.csv) ###")
print("\nLiczba brakujących wartości (NaN) w każdej kolumnie:")
nan_count_df = df.isna().sum()
print(nan_count_df)

print(f"\nCałkowita liczba brakujących wartości: {nan_count_df.sum()}")
print(f"Procent brakujących wartości: {(nan_count_df.sum() / df.size) * 100:.2f}%")
print(f"Liczba rekordów: {len(df)}")
print(f"Liczba kolumn: {len(df.columns)}")

# Sprawdzenie, które wiersze zawierają co najmniej jedno NaN
rows_with_nan_df = df[df.isna().any(axis=1)]
print(f"\nLiczba wierszy z co najmniej jednym NaN: {len(rows_with_nan_df)}")
if not rows_with_nan_df.empty:
    print("\nPrzykładowe wiersze z NaN (pierwsze 5):")
    print(rows_with_nan_df.head())

# Sprawdzenie brakujących wartości w skróconym zbiorze danych (df2)
print("\n### Analiza brakujących wartości w skróconym zbiorze danych (short_database.csv) ###")
print("\nLiczba brakujących wartości (NaN) w każdej kolumnie:")
nan_count_df2 = df2.isna().sum()
print(nan_count_df2)

print(f"\nCałkowita liczba brakujących wartości: {nan_count_df2.sum()}")
print(f"Procent brakujących wartości: {(nan_count_df2.sum() / df2.size) * 100:.2f}%")
print(f"Liczba rekordów: {len(df2)}")
print(f"Liczba kolumn: {len(df2.columns)}")

# Sprawdzenie, które wiersze zawierają co najmniej jedno NaN
rows_with_nan_df2 = df2[df2.isna().any(axis=1)]
print(f"\nLiczba wierszy z co najmniej jednym NaN: {len(rows_with_nan_df2)}")
if not rows_with_nan_df2.empty:
    print("\nPrzykładowe wiersze z NaN (pierwsze 5):")
    print(rows_with_nan_df2.head())

# Definicja kombinacji parametrów Prophet
param_combinations = [
    {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 1.0, 'holidays_prior_scale': 5.0, 'seasonality_mode': 'additive'},
    {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 5.0, 'holidays_prior_scale': 10.0, 'seasonality_mode': 'additive'},
    {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'holidays_prior_scale': 15.0, 'seasonality_mode': 'additive'},
    {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 15.0, 'holidays_prior_scale': 10.0, 'seasonality_mode': 'multiplicative'},
    {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 50.0, 'holidays_prior_scale': 0.01, 'seasonality_mode': 'multiplicative'},
    {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 50.0, 'holidays_prior_scale': 0.01, 'seasonality_mode': 'additive'}
]
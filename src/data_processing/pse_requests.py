import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import os

# Parametry
base_url = "https://api.raporty.pse.pl"
entity = "his-obc"
start_date = datetime(2016, 1, 1)
end_date = datetime(2024, 12, 31)
output_file = "../../data/other_data/his-obc_2016-2024.csv"

# Lista do przechowywania danych
all_data = []

# Generowanie zakresu dat
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime("%Y-%m-%d")
    
    # Utwórz URL z parametrem $filter
    filter_param = f"doba eq '{date_str}'"
    url = f"{base_url}/api/{entity}?$filter={filter_param}"

    # Wykonaj żądanie GET
    print(f"Pobieram dane dla daty: {date_str}")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Sprawdź, czy żądanie zakończyło się sukcesem

        # Parsuj odpowiedź jako JSON
        data = response.json()

        # Sprawdź, czy klucz "value" istnieje
        if "value" not in data:
            print(f"Brak klucza 'value' w odpowiedzi dla daty {date_str}:")
            print(json.dumps(data, indent=4, ensure_ascii=False))
            continue

        # Przetwarzaj dane z klucza "value"
        for record in data["value"]:
            # Mapuj pola JSON na kolumny
            mapped_record = {
                "doba": record.get("doba"),
                "udtczas": record.get("udtczas"),
                "obciazenie": record.get("obciazenie"),
                "udtczas_oreb": record.get("udtczas_oreb"),
                "business_date": record.get("business_date"),
                "source_datetime": record.get("source_datetime")
            }
            all_data.append(mapped_record)

    except requests.exceptions.HTTPError as http_err:
        print(f"Błąd HTTP dla daty {date_str}: {http_err}")
        print(f"Kod odpowiedzi: {response.status_code}")
        print(f"Treść odpowiedzi: {response.text}")
    except requests.exceptions.RequestException as err:
        print(f"Błąd żądania dla daty {date_str}: {err}")
    except (ValueError, KeyError) as err:
        print(f"Błąd parsowania danych dla daty {date_str}: {err}")
        print(f"Treść odpowiedzi: {response.text}")

    # Przejdź do następnej daty
    current_date += timedelta(days=1)

    # Opóźnienie między żądaniami (1 sekunda)
    time.sleep(0.05)

# Przekształć dane w DataFrame
if not all_data:
    print("Brak danych do zapisania. Kończę działanie.")
    exit()

df = pd.DataFrame(all_data)

# Sortuj dane po udtczas (timestamp)
df["udtczas"] = pd.to_datetime(df["udtczas"], format="%Y-%m-%d %H:%M")
df = df.sort_values("udtczas")

# Zapisz do pliku CSV
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"Dane zapisane do pliku: {output_file}")
print(f"Liczba rekordów: {len(df)}")
print(f"Kolumny w pliku: {df.columns.tolist()}")

# Wyświetl pierwsze 5 wierszy
print("\nPierwsze 5 wierszy danych:")
print(df.head())
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.dates as mdates

# Wczytaj dane 
df = pd.read_csv("../../data/database.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# KORELACJA ZMIENNYCH Z fixing_i_price
features = [
    "temp_waw", "wind_speed_waw", "cloud_cover_waw", "solar_radiation_waw",
    "temp_ksz", "wind_speed_ksz", "cloud_cover_ksz", "solar_radiation_ksz",
    "temp_krk", "wind_speed_krk", "cloud_cover_krk", "solar_radiation_krk",
    "temp_bab", "wind_speed_bab", "cloud_cover_bab", "solar_radiation_bab",
    "power_loss", "Network_loss",
    "Niemcy Bilans", "Czechy Bilans", "Litwa Bilans", "Słowacja Bilans", "Szwecja Bilans", "Ukraina Bilans",
    "hard_coal", "coal-derived", "lignite", "gas", "oil", "biomass", "wind_onshore", "solar",
    "fixing_i_volume", "Load", "gas_price", "gas_volume", "coal_pscmi1_pln_per_gj", "co2_price",
    "pln_usd", "brent_price", "day_of_week", "month", "hour", "fixing_i_price_lag24", "fixing_i_price_lag168", "is_holiday"
]
target = "fixing_i_price"

# Konwersja zmiennych logicznych/kategorycznych na numeryczne (np. is_holiday)
df["is_holiday"] = df["is_holiday"].astype(int)

# Usunięcie brakujących wartości (NaN) z wybranych kolumn
analysis_df = df[features + [target]].dropna()

# Oblicz korelacje
correlation_df = analysis_df.corr()[[target]].drop(target)
correlation_df.columns = ["Korelacja"]
correlation_df = correlation_df.sort_values(by="Korelacja", ascending=False)

# Ustawienie stylu wykresu (opcjonalne, można pominąć)
# plt.style.use("seaborn-v0_8")

# Tworzenie wykresu
plt.figure(figsize=(12, 8))

# Kolory: dodatnie korelacje na zielono, ujemne na czerwono
colors = ["#2ecc71" if x >= 0 else "#e74c3c" for x in correlation_df["Korelacja"]]

# Wykres słupkowy
sns.barplot(x="Korelacja", y=correlation_df.index, palette=colors, data=correlation_df)

# Dodaj etykiety i tytuł
plt.title("Korelacja zmiennych z fixing_i_price", fontsize=16, pad=20)
plt.xlabel("Współczynnik korelacji", fontsize=12)
plt.ylabel("Zmienna", fontsize=12)

# Dodaj siatkę dla lepszej czytelności
plt.grid(True, axis="x", linestyle="--", alpha=0.7)

# Dopasuj układ
plt.tight_layout()

# Zapisz wykres
plt.savefig("../../plots/correlation_with_fixing_i_price.png", dpi=300)
plt.close()

print("Wykres zapisany w ../../plots/correlation_with_fixing_i_price.png")

# Wyświetlenie wyników analizy korelacji w konsoli
print("Analiza korelacji zmiennych z fixing_i_price:")
print(correlation_df)

# Przekształcenie daty na kwartały
df["quarter"] = df["timestamp"].dt.to_period("Q").astype(str)

# Obliczenie średniej ceny dla każdego kwartału
quarterly_data = df.groupby("quarter")["fixing_i_price"].mean().reset_index()

# Tworzenie wykresu
plt.figure(figsize=(12, 6))
plt.plot(quarterly_data["quarter"], quarterly_data["fixing_i_price"], marker="o", color="#3498db", linewidth=1, markersize=5)
plt.title("Średnia cena fixing_i_price w ujęciu kwartalnym (2016–2024)", fontsize=14)
plt.xlabel("Kwartał", fontsize=12)
plt.ylabel("Średnia cena (PLN/MWh)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)

# Formatowanie osi X, aby etykiety były czytelne
plt.xticks(rotation=45, ha="right")
plt.gca().set_xticks(range(0, len(quarterly_data), 2))  # Pokazuj co drugi kwartał, aby uniknąć nakładania
plt.tight_layout()

# Zapis wykresu
plt.savefig("C:/mgr/EPF-Thesis/plots/quarterly_fixing_i_price.png", dpi=300)
plt.close()


# Wykres ceny fixing_i_price w dni robocze i święta
plt.figure(figsize=(14, 8))
# Dodaj etykiety i tytuł
# Grupowanie danych po dniu tygodnia i święcie, obliczanie średniej ceny
df["day_or_holiday"] = df.apply(lambda row: "Święto" if row["is_holiday"] == 1 else row["timestamp"].day_name(), axis=1)
day_or_holiday_avg_price = df.groupby("day_or_holiday")[target].mean().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Święto"]
)

# Wykres słupkowy
sns.barplot(x=day_or_holiday_avg_price.index, y=day_or_holiday_avg_price.values, palette="viridis")

# Dodaj etykiety i tytuł
plt.title("Średnia cena fixing_i_price w dni tygodnia i święta [PLN / MWh]", fontsize=16, pad=20)
plt.xlabel("Dzień tygodnia lub święto", fontsize=12)
plt.ylabel("Średnia cena fixing_i_price", fontsize=12)

# Dodaj wartości na słupkach
for index, value in enumerate(day_or_holiday_avg_price.values):
    plt.text(index, value + 0.5, f"{value:.2f}", ha="center", fontsize=10)

# Dodaj siatkę dla lepszej czytelności
plt.grid(True, axis="y", linestyle="--", alpha=0.7)

# Dopasuj układ
plt.tight_layout()

# Zapisz wykres
plt.savefig("../../plots/fixing_i_price_weekdays_holidays.png", dpi=300)
plt.close()

print("Wykres zapisany w ../../plots/fixing_i_price_weekdays_holidays.png")

# Utworzenie folderu energy w folderze plots, jeśli nie istnieje
output_dir = "C:/mgr/EPF-Thesis/plots/energy"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Wykresy dla całego okresu (2016–2024) w ujęciu miesięcznym
# Agregacja danych do średnich miesięcznych
df["year_month"] = df["timestamp"].dt.to_period("M")
monthly_data = df.groupby("year_month").mean(numeric_only=True).reset_index()
monthly_data["year_month"] = monthly_data["year_month"].astype(str)

# Wykres obszarowy dla produkcji energii (cały okres)
plt.figure(figsize=(12, 6))
plt.stackplot(
    monthly_data["year_month"],
    monthly_data["hard_coal"],
    monthly_data["coal-derived"],
    monthly_data["lignite"],
    monthly_data["gas"],
    monthly_data["oil"],
    monthly_data["biomass"],
    monthly_data["wind_onshore"],
    monthly_data["solar"],
    labels=["Węgiel kamienny", "Paliwa pochodne węgla", "Węgiel brunatny", "Gaz", "Ropa", "Biomasa", "Farmy wiatrowe", "Słońce"],
    colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
)
plt.title("Średnia miesięczna produkcja energii z różnych źródeł (2016–2024)", fontsize=14)
plt.xlabel("Rok-Miesiąc", fontsize=12)
plt.ylabel("Produkcja energii (MW)", fontsize=12)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(rotation=45, ha="right")
plt.gca().set_xticks(range(0, len(monthly_data), 6))  # Pokazuj co 6. miesiąc dla czytelności
plt.tight_layout()
plt.savefig(f"{output_dir}/energy_production_time_series_full.png", dpi=300, bbox_inches="tight")
plt.close()

# Wykres liniowy dla zapotrzebowania (cały okres)
plt.figure(figsize=(12, 6))
plt.plot(monthly_data["year_month"], monthly_data["Load"], color="#3498db", label="Zapotrzebowanie")
plt.title("Średnie miesięczne zapotrzebowanie na energię (2016–2024)", fontsize=14)
plt.xlabel("Rok-Miesiąc", fontsize=12)
plt.ylabel("Zapotrzebowanie (MW)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(rotation=45, ha="right")
plt.gca().set_xticks(range(0, len(monthly_data), 6))
plt.tight_layout()
plt.savefig(f"{output_dir}/load_time_series_full.png", dpi=300)
plt.close()

# 2. Wykresy dla roku 2022 w ujęciu dziennym
# Filtrowanie danych dla roku 2022
data_2022 = df[df["timestamp"].dt.year == 2022]
data_2022["day"] = data_2022["timestamp"].dt.date
daily_data_2022 = data_2022.groupby("day").mean(numeric_only=True).reset_index()
daily_data_2022["day"] = pd.to_datetime(daily_data_2022["day"])

# Wykres obszarowy dla produkcji energii (2022)
plt.figure(figsize=(12, 6))
plt.stackplot(
    daily_data_2022["day"],
    daily_data_2022["hard_coal"],
    daily_data_2022["coal-derived"],
    daily_data_2022["lignite"],
    daily_data_2022["gas"],
    daily_data_2022["oil"],
    daily_data_2022["biomass"],
    daily_data_2022["wind_onshore"],
    daily_data_2022["solar"],
    labels=["Węgiel kamienny", "Paliwa pochodne węgla", "Węgiel brunatny", "Gaz", "Ropa", "Biomasa", "Farmy wiatrowe", "Słońce"],
    colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
)
plt.title("Średnia dzienna produkcja energii z różnych źródeł w 2022 roku", fontsize=14)
plt.xlabel("df", fontsize=12)
plt.ylabel("Produkcja energii (MW)", fontsize=12)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(rotation=45, ha="right")
plt.gca().set_xticks(daily_data_2022["day"][::30])  # Pokazuj co 30. dzień dla czytelności
plt.tight_layout()
plt.savefig(f"{output_dir}/energy_production_time_series_2022.png", dpi=300, bbox_inches="tight")
plt.close()

# Wykres liniowy dla zapotrzebowania (2022)
plt.figure(figsize=(12, 6))
plt.plot(daily_data_2022["day"], daily_data_2022["Load"], color="#3498db", label="Zapotrzebowanie")
plt.title("Średnie dzienne zapotrzebowanie na energię w 2022 roku", fontsize=14)
plt.xlabel("df", fontsize=12)
plt.ylabel("Zapotrzebowanie (MW)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(rotation=45, ha="right")
plt.gca().set_xticks(daily_data_2022["day"][::30])
plt.tight_layout()
plt.savefig(f"{output_dir}/load_time_series_2022.png", dpi=300)
plt.close()


# Utworzenie folderu cross_border w folderze plots, jeśli nie istnieje
output_dir = "C:/mgr/EPF-Thesis/plots/cross_border"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Zakładam, że db to już wczytany DataFrame
data = df.copy()

# Dodanie kolumny z rokiem
data["year"] = data["timestamp"].dt.year

# Kolumny wymiany transgranicznej (wartości dodatnie = eksport, ujemne = import)
countries = {
    "Niemcy Bilans": "Niemcy",
    "Czechy Bilans": "Czechy",
    "Litwa Bilans": "Litwa",
    "Słowacja Bilans": "Słowacja",
    "Szwecja Bilans": "Szwecja",
    "Ukraina Bilans": "Ukraina"
}

# Obliczenie średniej rocznej wymiany dla każdego kraju i roku
annual_exchange = data.groupby("year")[[col for col in countries.keys()]].mean().reset_index()

# Wypisanie wartości na konsoli
print("Średnia roczna wymiana transgraniczna (w MW):")
for year in annual_exchange["year"]:
    print(f"\nRok {year}:")
    for col in countries.keys():
        value = annual_exchange.loc[annual_exchange["year"] == year, col].iloc[0]
        print(f"{countries[col]}: {value:.2f} MW")

# Wykres liniowy dla salda wymiany transgranicznej w latach 2016–2024
plt.figure(figsize=(12, 6))
for col, country in countries.items():
    plt.plot(annual_exchange["year"], annual_exchange[col], label=country, linewidth=2)
plt.axhline(0, color="black", linestyle="--", alpha=0.5)
plt.title("Roczne saldo wymiany transgranicznej energii elektrycznej (2016–2024)", fontsize=14)
plt.xlabel("Rok", fontsize=12)
plt.ylabel("Saldo wymiany (MW)", fontsize=12)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(f"{output_dir}/cross_border_balance_2016_2024.png", dpi=300, bbox_inches="tight")
plt.close()

# Utworzenie folderu fuels w folderze plots, jeśli nie istnieje
output_dir = "C:/mgr/EPF-Thesis/plots/fuels"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Agregacja danych do średnich miesięcznych
data["year_month"] = data["timestamp"].dt.to_period("M")
monthly_data = data.groupby("year_month")[["gas_price", "co2_price", "coal_pscmi1_pln_per_gj", "brent_price"]].mean().reset_index()
monthly_data["year_month"] = monthly_data["year_month"].dt.to_timestamp()

# Wykres liniowy dla cen paliw i emisji CO2 w skali logarytmicznej
fig, ax1 = plt.subplots(figsize=(12, 6))

# Oś pierwsza (dla gas_price, coal_pscmi1_pln_per_gj, brent_price) w skali logarytmicznej
ax1.plot(monthly_data["year_month"], monthly_data["gas_price"], label="Cena gazu (PLN/MWh)", color="blue", linewidth=2)
ax1.plot(monthly_data["year_month"], monthly_data["co2_price"], label="Cena CO$_2$ (PLN/tCO$_2$)", color="red", linewidth=2)
ax1.plot(monthly_data["year_month"], monthly_data["brent_price"], label="Cena ropy Brent (PLN/bar)", color="green", linewidth=2)
ax1.set_xlabel("Rok", fontsize=12)
ax1.set_ylabel("Cena (PLN/MWh, PLN/tCO$_2$, PLN/bar)", fontsize=12)
ax1.set_yscale("log")  # Skala logarytmiczna dla osi Y
ax1.legend(loc="upper left")
ax1.grid(True, linestyle="--", alpha=0.7)

# Oś druga (dla coal_pscmi1_pln_per_gj) w skali logarytmicznej
ax2 = ax1.twinx()
ax2.plot(monthly_data["year_month"], monthly_data["coal_pscmi1_pln_per_gj"], label="Cena węgla (PLN/GJ)", color="black", linewidth=2)
ax2.set_ylabel("Cena węgla (PLN/GJ)", fontsize=12)
ax2.set_yscale("log")  # Skala logarytmiczna dla osi Y
ax2.legend(loc="upper right")

# Ustawienie etykiet na osi X (co roku, tylko rok)
ax1.xaxis.set_major_locator(mdates.YearLocator())  # Etykiety co roku
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # Format: tylko rok (np. 2016)
plt.xticks(rotation=45, ha="right")

plt.title("Ceny paliw kopalnych i emisji CO$_2$ (2016–2024) w skali logarytmicznej", fontsize=14)
plt.tight_layout()
plt.savefig(f"{output_dir}/fuel_prices_2016_2024.png", dpi=300, bbox_inches="tight")
plt.close()

# Utworzenie folderu losses w folderze plots, jeśli nie istnieje
output_dir = "C:/mgr/EPF-Thesis/plots/losses"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Agregacja danych do średnich miesięcznych
data["year_month"] = data["timestamp"].dt.to_period("M")
monthly_data = data.groupby("year_month")[["power_loss", "Network_loss"]].mean().reset_index()
monthly_data["year_month"] = monthly_data["year_month"].dt.to_timestamp()

# Wykres liniowy dla strat mocy
plt.figure(figsize=(12, 6))
plt.plot(monthly_data["year_month"], monthly_data["power_loss"], label="Straty mocy stacji elektrycznych (MW)", color="red", linewidth=2)
plt.plot(monthly_data["year_month"], monthly_data["Network_loss"], label="Utrata mocy w sieci (MW)", color="blue", linewidth=2)

# Ustawienie etykiet na osi X (co roku, tylko rok)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Etykiety co roku
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # Format: tylko rok (np. 2016)
plt.xticks(rotation=45, ha="right")

plt.xlabel("Rok", fontsize=12)
plt.ylabel("Straty mocy (MW)", fontsize=12)
plt.title("Straty mocy w wyniku awarii i w sieci (2016–2024)", fontsize=14)
plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(f"{output_dir}/power_losses_2016_2024.png", dpi=300, bbox_inches="tight")
plt.close()

# Filtrowanie danych dla okresu od marca 2024 do grudnia 2024
data["date"] = data["timestamp"].dt.date
start_date = pd.to_datetime("2024-03-01")
end_date = pd.to_datetime("2024-12-31")
filtered_data = data[(data["timestamp"] >= start_date) & (data["timestamp"] <= end_date)]

# Agregacja danych do średnich dziennych
daily_data = filtered_data.groupby("date")[["Network_loss"]].mean().reset_index()
daily_data["date"] = pd.to_datetime(daily_data["date"])

# Wykres liniowy dla network_loss w 2024 roku
plt.figure(figsize=(12, 6))
plt.plot(daily_data["date"], daily_data["Network_loss"], label="Utrata mocy w sieci (MW)", color="blue", linewidth=2)

# Ustawienie etykiet na osi X (co miesiąc)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Etykiety co miesiąc
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # Format: rok-miesiąc (np. 2024-03)
plt.xticks(rotation=45, ha="right")

plt.xlabel("Data", fontsize=12)
plt.ylabel("Straty mocy w sieci (MW)", fontsize=12)
plt.title("Straty mocy w sieci (marzec 2024 – grudzień 2024)", fontsize=14)
plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(f"{output_dir}/network_loss_2024.png", dpi=300, bbox_inches="tight")
plt.close()

# Wykres dla Load i fixing_i_volume w ujęciu miesięcznym (2016–2024)
# Utworzenie folderu market w folderze plots, jeśli nie istnieje
output_dir = "C:/mgr/EPF-Thesis/plots/market"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Agregacja danych do średnich miesięcznych
data["year_month"] = data["timestamp"].dt.to_period("M")
monthly_data = data.groupby("year_month")[["Load", "fixing_i_volume"]].mean().reset_index()
monthly_data["year_month"] = monthly_data["year_month"].dt.to_timestamp()

# Wykres liniowy dla Load i fixing_i_volume
fig, ax1 = plt.subplots(figsize=(12, 6))

# Oś pierwsza (dla Load)
ax1.plot(monthly_data["year_month"], monthly_data["Load"], label="Zapotrzebowanie (MW)", color="blue", linewidth=2)
ax1.set_xlabel("Rok", fontsize=12)
ax1.set_ylabel("Zapotrzebowanie (MW)", fontsize=12, color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.grid(True, linestyle="--", alpha=0.7)

# Oś druga (dla fixing_i_volume)
ax2 = ax1.twinx()
ax2.plot(monthly_data["year_month"], monthly_data["fixing_i_volume"], label="Wolumen sprzedaży na RDN (MWh)", color="red", linewidth=2)
ax2.set_ylabel("Wolumen sprzedaży (MWh)", fontsize=12, color="red")
ax2.tick_params(axis="y", labelcolor="red")

# Ustawienie etykiet na osi X (co roku)
ax1.xaxis.set_major_locator(mdates.YearLocator())  # Etykiety co roku
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # Format: tylko rok (np. 2016)
plt.xticks(rotation=45, ha="right")

# Tytuł i legenda
plt.title("Średnie miesięczne zapotrzebowanie i wolumen sprzedaży na RDN (2016–2024)", fontsize=14)
fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.tight_layout()
plt.savefig(f"{output_dir}/load_vs_volume_2016_2024.png", dpi=300, bbox_inches="tight")
plt.close()

# Grupowanie danych według roku i obliczanie średniej wartości pln_usd
annual_pln_usd = data.groupby("year")["pln_usd"].mean().reset_index()

# Wyświetlenie wyników na konsoli
print("Średnioroczny kurs PLN/USD w latach 2016-2024:")
for index, row in annual_pln_usd.iterrows():
    print(f"Rok: {int(row['year'])}, Średni kurs PLN/USD: {row['pln_usd']:.2f}")

# # Obliczenie macierzy korelacji
# correlation_matrix = data.corr()
# # Wyodrębnienie korelacji dla fixing_i_price
# fixing_i_price_correlations = correlation_matrix['fixing_i_price'].drop('fixing_i_price')
# # Wyświetlenie korelacji dla fixing_i_price
# print("Korelacje zmiennej fixing_i_price z pozostałymi zmiennymi:")
# print(fixing_i_price_correlations)

# # Wizualizacja macierzy korelacji za pomocą heatmapy
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
# plt.title('Macierz korelacji zmiennych')
# plt.show()


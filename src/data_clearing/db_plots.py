import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Wczytaj dane
df = pd.read_csv("../../data/database.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Wybierz zmienne do analizy korelacji (wszystkie zmienne numeryczne poza timestamp)
features = [
    "temp_waw", "wind_speed_waw", "cloud_cover_waw", "solar_radiation_waw",
    "temp_ksz", "wind_speed_ksz", "cloud_cover_ksz", "solar_radiation_ksz",
    "temp_krk", "wind_speed_krk", "cloud_cover_krk", "solar_radiation_krk",
    "temp_bab", "wind_speed_bab", "cloud_cover_bab", "solar_radiation_bab",
    "power_loss", "Network_loss",
    "Niemcy Bilans", "Czechy Bilans", "Litwa Bilans", "Słowacja Bilans", "Szwecja Bilans", "Ukraina Bilans",
    "hard_coal", "coal-derived", "lignite", "gas", "oil", "biomass", "wind_onshore", "solar",
    "fixing_i_volume", "Load", "gas_price", "gas_volume", "coal_pscmi1_pln_per_gj", "co2_price",
    "pln_usd", "brent_price", "day_of_week", "is_holiday"
]
target = "fixing_i_price"

# # Oblicz korelacje
# correlation_df = df[features + [target]].corr()[[target]].drop(target)
# correlation_df.columns = ["Korelacja"]
# correlation_df = correlation_df.sort_values(by="Korelacja", ascending=False)

# # Ustaw styl wykresu
# # plt.style.use("seaborn")
# plt.figure(figsize=(12, 8))

# # Kolory: dodatnie korelacje na zielono, ujemne na czerwono
# colors = ["#2ecc71" if x >= 0 else "#e74c3c" for x in correlation_df["Korelacja"]]

# # Wykres słupkowy
# sns.barplot(x="Korelacja", y=correlation_df.index, palette=colors, df=correlation_df)

# # Dodaj etykiety i tytuł
# plt.title("Korelacja zmiennych z fixing_i_price", fontsize=16, pad=20)
# plt.xlabel("Współczynnik korelacji", fontsize=12)
# plt.ylabel("Zmienna", fontsize=12)
# plt.xticks(ticks=range(0, len(correlation_df.index), 3), labels=correlation_df.index[::3])

# # Dodaj siatkę dla lepszej czytelności
# plt.grid(True, axis="x", linestyle="--", alpha=0.7)
# # Dopasuj układ
# plt.tight_layout()

# # Zapisz wykres
# plt.savefig("../../plots/correlation_with_fixing_i_price.png", dpi=300)
# plt.close()

# print("Wykres zapisany w ../../plots/correlation_with_fixing_i_price.png")

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
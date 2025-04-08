import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Oblicz korelacje
correlation_df = df[features + [target]].corr()[[target]].drop(target)
correlation_df.columns = ["Korelacja"]
correlation_df = correlation_df.sort_values(by="Korelacja", ascending=False)

# Ustaw styl wykresu
# plt.style.use("seaborn")
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

# Wykres ceny fixing_i_price przez lata
plt.figure(figsize=(14, 8))

# Grupowanie danych po roku i obliczanie średniej ceny
df["year"] = df["timestamp"].dt.year
yearly_avg_price = df.groupby("year")[target].mean()

# Wykres liniowy
plt.plot(yearly_avg_price.index, yearly_avg_price.values, marker="o", linestyle="-", color="#3498db")

# Dodaj etykiety i tytuł
plt.title("Średnia cena fixing_i_price przez lata", fontsize=16, pad=20)
plt.xlabel("Rok", fontsize=12)
plt.ylabel("Średnia cena fixing_i_price", fontsize=12)

# Dodaj siatkę dla lepszej czytelności
plt.grid(True, linestyle="--", alpha=0.7)

# Dopasuj układ
plt.tight_layout()

# Zapisz wykres
plt.savefig("../../plots/fixing_i_price_over_years.png", dpi=300)
plt.close()
print("Wykres zapisany w ../../plots/fixing_i_price_over_years.png")

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
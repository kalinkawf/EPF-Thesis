import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.dates as mdates
import mplfinance as mpf

# Wczytaj dane
df = pd.read_csv("../../data/database.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

df_short = pd.read_csv("../../data/short_database.csv")
df_short["timestamp"] = pd.to_datetime(df_short["timestamp"])

# Lista zmiennych
features = [
    "temp_waw", "wind_speed_waw", "cloud_cover_waw", "solar_radiation_waw",
    "temp_ksz", "wind_speed_ksz", "cloud_cover_ksz", "solar_radiation_ksz",
    "temp_krk", "wind_speed_krk", "cloud_cover_krk", "solar_radiation_krk",
    "temp_bab", "wind_speed_bab", "cloud_cover_bab", "solar_radiation_bab",
    "power_loss", "Network_loss",
    "Niemcy Bilans", "Czechy Bilans", "Litwa Bilans", "Słowacja Bilans", "Szwecja Bilans", "Ukraina Bilans",
    "hard_coal", "coal-derived", "lignite", "gas", "oil", "biomass", "wind_onshore", "solar",
    "fixing_i_volume", "Load", "non_emissive_sources_percentage",
    "gas_price", "coal_pscmi1_pln_per_gj", "co2_price", "pln_usd", "brent_price",
    "day_of_week", "month", "hour",
    "fixing_i_price_mean24", "fixing_i_price_mean48",
    "fixing_i_price_lag24", "fixing_i_price_lag48", "fixing_i_price_lag72",
    "fixing_i_price_lag96", "fixing_i_price_lag120", "fixing_i_price_lag144", "fixing_i_price_lag168",
    "is_holiday", "peak_hour", "RB_price", "se_price", "sk_price", "cz_price", "lt_price", "pln_eur",
]
target = "fixing_i_price"

# Zaktualizowana lista zmiennych nieliniowych (różnica > 0,1)
non_linear_features = [
    "co2_price", "coal_pscmi1_pln_per_gj", "pln_usd",
    "solar", "gas", "biomass", "coal-derived",
    "solar_radiation_bab", "solar_radiation_ksz", "solar_radiation_krk", "solar_radiation_waw"
]

dataset_short = [
    "fixing_i_price_lag24",
    "gas_price", "co2_price", "brent_price", "pln_usd", "coal_pscmi1_pln_per_gj",
    "power_loss", "fixing_i_volume", "solar", "gas", "oil", "Load",  "avg_wind_speed", "avg_solar_radiation",
    "hour", "month", "is_holiday", "non_emissive_sources_percentage", "day_of_week", "RB_price", "se_price",
    "sk_price", "cz_price", "lt_price", "pln_eur",
]

# Przekształcenie daty na kwartały
df["quarter"] = df["timestamp"].dt.to_period("Q").astype(str)

# Obliczenie średniej ceny dla każdego kwartału
quarterly_data = df.groupby("quarter")["fixing_i_price"].mean().reset_index()

# Tworzenie wykresu
plt.figure(figsize=(12, 6))
plt.plot(quarterly_data["quarter"], quarterly_data["fixing_i_price"], marker="o", color="#3498db", linewidth=1, markersize=5)
plt.title("Średnia cena fixing_i_price w ujęciu kwartalnym (2016–2023)", fontsize=14)
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

# wykres świecowy
df.set_index("timestamp", inplace=True)
ohlc = df["fixing_i_price"].resample("ME").agg({
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last"
})

ohlc.index.name = "Data"

print(mpf.available_styles())

fig, axlist = mpf.plot(
    ohlc,
    type='candle',
    style='yahoo',
    title='Fixing_i_price - wykres świecowy (miesięczny)',
    ylabel='Cena (PLN/MWh)',
    figratio=(16, 6),
    figscale=1.2,
    tight_layout=True,  # <--- to usuwa marginesy
    datetime_format='%m.%Y',
    returnfig=True,
)

# Dodaj tytuł nad wykresem (nie w środku)
fig.suptitle(
    'Fixing_i_price - wykres świecowy (miesięczny)',
    fontsize=14,
    fontweight='normal',  # <-- usuwa pogrubienie
    y=0.98                # <-- wyżej niż domyślnie
)
main_ax = axlist[0]

# 🔧 Formatowanie osi Y
main_ax.yaxis.set_label_position("left")  # etykieta po prawej
main_ax.yaxis.tick_left()                # cyfry po prawej

fig.savefig('C:/mgr/EPF-Thesis/plots/candlestick_fixing_i_price.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# KORELACJA
# # Oblicz korelację Pearsona dla zmiennych liniowych
# pearson_features = [f for f in features if f not in non_linear_features]
# pearson_corr = df[pearson_features + [target]].corr(method="pearson")[target].drop(target)

# # Oblicz korelację Spearmana dla zmiennych nieliniowych
# spearman_corr = df[non_linear_features + [target]].corr(method="spearman")[target].drop(target)

# # Połącz wyniki
# correlation_df = pd.concat([pearson_corr, spearman_corr], axis=0).to_frame()
# correlation_df.columns = ["Korelacja"]
# correlation_df = correlation_df.sort_values(by="Korelacja", ascending=False)

# # Tworzenie wykresu korelacji
# plt.figure(figsize=(12, 10))

# # Kolory: gradient w zależności od wartości korelacji
# colors = sns.diverging_palette(10, 130, as_cmap=True)
# colors = [colors(x) for x in (correlation_df["Korelacja"] + 1) / 2]  # Skalowanie do [0, 1]

# # Wykres słupkowy
# ax = sns.barplot(x="Korelacja", y=correlation_df.index, palette=colors, data=correlation_df)

# # Dodaj wartości korelacji na słupkach
# for i, v in enumerate(correlation_df["Korelacja"]):
#     ax.text(v if v >= 0 else v - 0.15, i, f"{v:.2f}", va="center", ha="right" if v < 0 else "left", fontsize=10)

# # Dodaj linie progu istotności
# plt.axvline(x=0.3, color="gray", linestyle="--", alpha=0.7, label="Próg istotności (|r| = 0.3)")
# plt.axvline(x=-0.3, color="gray", linestyle="--", alpha=0.7)

# # Dodaj etykiety i tytuł
# plt.title("Korelacja zmiennych z fixing_i_price (Pearson i Spearman)", fontsize=16, pad=20)
# plt.xlabel("Współczynnik korelacji", fontsize=12)
# plt.ylabel("Zmienna", fontsize=12)

# # Dodaj siatkę i legendę
# plt.grid(True, axis="x", linestyle="--", alpha=0.7)
# plt.legend()

# # Dopasuj układ
# plt.tight_layout()

# # Zapisz wykres
# plt.savefig("../../plots/correlation_with_fixing_i_price.png", dpi=300)
# plt.close()

# # Oblicz macierz korelacji między wybranymi zmiennymi objaśniającymi (używamy Pearsona dla uproszczenia)
# corr_matrix = df_short[dataset_short].corr(method="pearson")

# # Tworzenie heatmapy
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0,
#             square=True, cbar_kws={"label": "Współczynnik korelacji"})
# plt.title("Heatmapa korelacji między zmiennymi objaśniającymi o największym wpływie na fixing_i_price", fontsize=16, pad=20)
# plt.xticks(rotation=45, ha="right")
# plt.yticks(rotation=0)

# # Dopasuj układ
# plt.tight_layout()

# # Zapisz heatmapę
# plt.savefig("../../plots/heatmap_short_db_features.png", dpi=300)
# plt.close()

# # Podział na okresy spokojny i niespokojny
# calm_period = df[(df["timestamp"] >= "2016-01-01") & (df["timestamp"] <= "2019-12-31")]
# volatile_period = df[(df["timestamp"] >= "2020-01-01") & (df["timestamp"] <= "2023-12-31")]

# # Oblicz statystyki dla obu okresów
# calm_stats = calm_period["fixing_i_price"].describe()[["mean", "std", "min", "max", "25%", "50%", "75%"]]
# volatile_stats = volatile_period["fixing_i_price"].describe()[["mean", "std", "min", "max", "25%", "50%", "75%"]]

# # Dodatkowe statystyki
# # Współczynnik zmienności (CV = std / mean * 100%)
# calm_cv = (calm_stats["std"] / calm_stats["mean"]) * 100
# volatile_cv = (volatile_stats["std"] / volatile_stats["mean"]) * 100

# # Procent dni z ceną powyżej 500 PLN/MWh
# calm_high_price = (calm_period["fixing_i_price"] > 500).mean() * 100
# volatile_high_price = (volatile_period["fixing_i_price"] > 500).mean() * 100

# # Wyświetl statystyki na konsoli
# print("\nStatystyki dla okresu spokojnego (2016–2019):")
# print(f"Średnia: {calm_stats['mean']:.2f} PLN/MWh")
# print(f"Mediana: {calm_stats['50%']:.2f} PLN/MWh")
# print(f"Odchylenie standardowe: {calm_stats['std']:.2f} PLN/MWh")
# print(f"Współczynnik zmienności: {calm_cv:.2f}%")
# print(f"Kwartyl Q1 (25%): {calm_stats['25%']:.2f} PLN/MWh")
# print(f"Kwartyl Q3 (75%): {calm_stats['75%']:.2f} PLN/MWh")
# print(f"Minimum: {calm_stats['min']:.2f} PLN/MWh")
# print(f"Maksimum: {calm_stats['max']:.2f} PLN/MWh")
# print(f"Procent dni z ceną powyżej 500 PLN/MWh: {calm_high_price:.2f}%")
# print()

# print("Statystyki dla okresu niespokojnego (2020–2023):")
# print(f"Średnia: {volatile_stats['mean']:.2f} PLN/MWh")
# print(f"Mediana: {volatile_stats['50%']:.2f} PLN/MWh")
# print(f"Odchylenie standardowe: {volatile_stats['std']:.2f} PLN/MWh")
# print(f"Współczynnik zmienności: {volatile_cv:.2f}%")
# print(f"Kwartyl Q1 (25%): {volatile_stats['25%']:.2f} PLN/MWh")
# print(f"Kwartyl Q3 (75%): {volatile_stats['75%']:.2f} PLN/MWh")
# print(f"Minimum: {volatile_stats['min']:.2f} PLN/MWh")
# print(f"Maksimum: {volatile_stats['max']:.2f} PLN/MWh")
# print(f"Procent dni z ceną powyżej 500 PLN/MWh: {volatile_high_price:.2f}%")
# print()

# # Definicja przedziałów dla zbiorów (proporcja 75/25)
# # Okres spokojny (2016–2019)
# calm_train_end = pd.to_datetime("2018-12-31 23:00:00")
# calm_test_end = pd.to_datetime("2019-12-31 23:00:00")

# # Okres niespokojny (2020–2023)
# volatile_train_end = pd.to_datetime("2022-12-31 23:00:00")
# volatile_test_end = pd.to_datetime("2023-12-31 23:00:00")

# # Tworzenie jednego wykresu z dwoma panelami (jeden nad drugim)
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# # Funkcja do rysowania jednego okresu na podanym panelu
# def plot_period(df, ax, period_name, train_end, test_end):
#     # Treningowy (niebieski)
#     train_data = df[df["timestamp"] <= train_end]
#     ax.plot(train_data["timestamp"], train_data["fixing_i_price"], color="blue", label="Zbiór treningowy", alpha=0.7)
    
#     # Testowy (zielony)
#     test_data = df[(df["timestamp"] > train_end) & (df["timestamp"] <= test_end)]
#     ax.plot(test_data["timestamp"], test_data["fixing_i_price"], color="green", label="Zbiór testowy", alpha=0.7)
    
#     # Dodaj etykiety i tytuł
#     ax.set_title(f"Szereg czasowy ceny energii ({period_name}) z podziałem na zbiory", fontsize=14, pad=15)
#     ax.set_ylabel("Cena energii (PLN/MWh)", fontsize=12)
    
#     # Dodaj siatkę i legendę
#     ax.grid(True, linestyle="--", alpha=0.7)
#     ax.legend()

# # Wygeneruj wykresy na obu panelach
# plot_period(calm_period, ax1, "Okres spokojny (2016–2019)", calm_train_end, calm_test_end)
# plot_period(volatile_period, ax2, "Okres niespokojny (2020–2023)", volatile_train_end, volatile_test_end)

# # Dodaj wspólną etykietę osi X
# ax2.set_xlabel("Czas", fontsize=12)

# # Dopasuj układ
# plt.tight_layout()

# # Zapisz wykres
# plt.savefig("../../plots/periods_split_combined.png", dpi=300)
# plt.close()

# # Wyświetlenie wyników analizy korelacji w konsoli
# print("Analiza korelacji zmiennych z fixing_i_price:")
# print(correlation_df)

# # Wykres ceny fixing_i_price w dni robocze i święta
# plt.figure(figsize=(14, 8))
# # Dodaj etykiety i tytuł
# # Grupowanie danych po dniu tygodnia i święcie, obliczanie średniej ceny
# df["day_or_holiday"] = df.apply(lambda row: "Święto" if row["is_holiday"] == 1 else row["timestamp"].day_name(), axis=1)
# day_or_holiday_avg_price = df.groupby("day_or_holiday")[target].mean().reindex(
#     ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Święto"]
# )

# # Wykres słupkowy
# sns.barplot(x=day_or_holiday_avg_price.index, y=day_or_holiday_avg_price.values, palette="viridis")

# # Dodaj etykiety i tytuł
# plt.title("Średnia cena fixing_i_price w dni tygodnia i święta [PLN / MWh]", fontsize=16, pad=20)
# plt.xlabel("Dzień tygodnia lub święto", fontsize=12)
# plt.ylabel("Średnia cena fixing_i_price", fontsize=12)

# # Dodaj wartości na słupkach
# for index, value in enumerate(day_or_holiday_avg_price.values):
#     plt.text(index, value + 0.5, f"{value:.2f}", ha="center", fontsize=10)

# # Dodaj siatkę dla lepszej czytelności
# plt.grid(True, axis="y", linestyle="--", alpha=0.7)

# # Dopasuj układ
# plt.tight_layout()

# # Zapisz wykres
# plt.savefig("../../plots/fixing_i_price_weekdays_holidays.png", dpi=300)
# plt.close()

# print("Wykres zapisany w ../../plots/fixing_i_price_weekdays_holidays.png")

# # Utworzenie folderu energy w folderze plots, jeśli nie istnieje
# output_dir = "C:/mgr/EPF-Thesis/plots/energy"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # 1. Wykresy dla całego okresu (2016-2023) w ujęciu miesięcznym
# # Agregacja danych do średnich miesięcznych
# df["year_month"] = df["timestamp"].dt.to_period("M")
# monthly_data = df.groupby("year_month").mean(numeric_only=True).reset_index()
# monthly_data["year_month"] = monthly_data["year_month"].astype(str)

# # Wykres obszarowy dla produkcji energii (cały okres)
# plt.figure(figsize=(12, 6))
# plt.stackplot(
#     monthly_data["year_month"],
#     monthly_data["hard_coal"],
#     monthly_data["coal-derived"],
#     monthly_data["lignite"],
#     monthly_data["gas"],
#     monthly_data["oil"],
#     monthly_data["biomass"],
#     monthly_data["wind_onshore"],
#     monthly_data["solar"],
#     labels=["Węgiel kamienny", "Paliwa pochodne węgla", "Węgiel brunatny", "Gaz", "Ropa", "Biomasa", "Farmy wiatrowe", "Słońce"],
#     colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
# )
# plt.title("Średnia miesięczna produkcja energii z różnych źródeł (2016-2023)", fontsize=14)
# plt.xlabel("Rok-Miesiąc", fontsize=12)
# plt.ylabel("Produkcja energii (MW)", fontsize=12)
# plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
# plt.grid(True, linestyle="--", alpha=0.7)
# plt.xticks(rotation=45, ha="right")
# plt.gca().set_xticks(range(0, len(monthly_data), 6))  # Pokazuj co 6. miesiąc dla czytelności
# plt.tight_layout()
# plt.savefig(f"{output_dir}/energy_production_time_series_full.png", dpi=300, bbox_inches="tight")
# plt.close()

# # Wykres liniowy dla zapotrzebowania (cały okres)
# plt.figure(figsize=(12, 6))
# plt.plot(monthly_data["year_month"], monthly_data["Load"], color="#3498db", label="Zapotrzebowanie")
# plt.title("Średnie miesięczne zapotrzebowanie na energię (2016-2023)", fontsize=14)
# plt.xlabel("Rok-Miesiąc", fontsize=12)
# plt.ylabel("Zapotrzebowanie (MW)", fontsize=12)
# plt.legend()
# plt.grid(True, linestyle="--", alpha=0.7)
# plt.xticks(rotation=45, ha="right")
# plt.gca().set_xticks(range(0, len(monthly_data), 6))
# plt.tight_layout()
# plt.savefig(f"{output_dir}/load_time_series_full.png", dpi=300)
# plt.close()

# # 2. Wykresy dla roku 2022 w ujęciu dziennym
# # Filtrowanie danych dla roku 2022
# data_2022 = df[df["timestamp"].dt.year == 2022]
# data_2022["day"] = data_2022["timestamp"].dt.date
# daily_data_2022 = data_2022.groupby("day").mean(numeric_only=True).reset_index()
# daily_data_2022["day"] = pd.to_datetime(daily_data_2022["day"])

# # Wykres obszarowy dla produkcji energii (2022)
# plt.figure(figsize=(12, 6))
# plt.stackplot(
#     daily_data_2022["day"],
#     daily_data_2022["hard_coal"],
#     daily_data_2022["coal-derived"],
#     daily_data_2022["lignite"],
#     daily_data_2022["gas"],
#     daily_data_2022["oil"],
#     daily_data_2022["biomass"],
#     daily_data_2022["wind_onshore"],
#     daily_data_2022["solar"],
#     labels=["Węgiel kamienny", "Paliwa pochodne węgla", "Węgiel brunatny", "Gaz", "Ropa", "Biomasa", "Farmy wiatrowe", "Słońce"],
#     colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
# )
# plt.title("Średnia dzienna produkcja energii z różnych źródeł w 2022 roku", fontsize=14)
# plt.xlabel("df", fontsize=12)
# plt.ylabel("Produkcja energii (MW)", fontsize=12)
# plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
# plt.grid(True, linestyle="--", alpha=0.7)
# plt.xticks(rotation=45, ha="right")
# plt.gca().set_xticks(daily_data_2022["day"][::30])  # Pokazuj co 30. dzień dla czytelności
# plt.tight_layout()
# plt.savefig(f"{output_dir}/energy_production_time_series_2022.png", dpi=300, bbox_inches="tight")
# plt.close()

# # Wykres liniowy dla zapotrzebowania (2022)
# plt.figure(figsize=(12, 6))
# plt.plot(daily_data_2022["day"], daily_data_2022["Load"], color="#3498db", label="Zapotrzebowanie")
# plt.title("Średnie dzienne zapotrzebowanie na energię w 2022 roku", fontsize=14)
# plt.xlabel("df", fontsize=12)
# plt.ylabel("Zapotrzebowanie (MW)", fontsize=12)
# plt.legend()
# plt.grid(True, linestyle="--", alpha=0.7)
# plt.xticks(rotation=45, ha="right")
# plt.gca().set_xticks(daily_data_2022["day"][::30])
# plt.tight_layout()
# plt.savefig(f"{output_dir}/load_time_series_2022.png", dpi=300)
# plt.close()


# # Utworzenie folderu cross_border w folderze plots, jeśli nie istnieje
# output_dir = "C:/mgr/EPF-Thesis/plots/cross_border"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Zakładam, że db to już wczytany DataFrame
# data = df.copy()

# # Dodanie kolumny z rokiem
# data["year"] = data["timestamp"].dt.year

# # Kolumny wymiany transgranicznej (wartości dodatnie = eksport, ujemne = import)
# countries = {
#     "Niemcy Bilans": "Niemcy",
#     "Czechy Bilans": "Czechy",
#     "Litwa Bilans": "Litwa",
#     "Słowacja Bilans": "Słowacja",
#     "Szwecja Bilans": "Szwecja",
#     "Ukraina Bilans": "Ukraina"
# }

# # Obliczenie średniej rocznej wymiany dla każdego kraju i roku
# annual_exchange = data.groupby("year")[[col for col in countries.keys()]].mean().reset_index()

# # Wypisanie wartości na konsoli
# print("Średnia roczna wymiana transgraniczna (w MW):")
# for year in annual_exchange["year"]:
#     print(f"\nRok {year}:")
#     for col in countries.keys():
#         value = annual_exchange.loc[annual_exchange["year"] == year, col].iloc[0]
#         print(f"{countries[col]}: {value:.2f} MW")

# # Wykres liniowy dla salda wymiany transgranicznej w latach 2016-2023
# plt.figure(figsize=(12, 6))
# for col, country in countries.items():
#     plt.plot(annual_exchange["year"], annual_exchange[col], label=country, linewidth=2)
# plt.axhline(0, color="black", linestyle="--", alpha=0.5)
# plt.title("Roczne saldo wymiany transgranicznej energii elektrycznej (2016-2023)", fontsize=14)
# plt.xlabel("Rok", fontsize=12)
# plt.ylabel("Saldo wymiany (MW)", fontsize=12)
# plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
# plt.grid(True, linestyle="--", alpha=0.7)
# plt.tight_layout()
# plt.savefig(f"{output_dir}/cross_border_balance_2016_2024.png", dpi=300, bbox_inches="tight")
# plt.close()

# # Utworzenie folderu fuels w folderze plots, jeśli nie istnieje
# output_dir = "C:/mgr/EPF-Thesis/plots/fuels"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Agregacja danych do średnich miesięcznych
# data["year_month"] = data["timestamp"].dt.to_period("M")
# monthly_data = data.groupby("year_month")[["gas_price", "co2_price", "coal_pscmi1_pln_per_gj", "brent_price"]].mean().reset_index()
# monthly_data["year_month"] = monthly_data["year_month"].dt.to_timestamp()

# # Wykres liniowy dla cen paliw i emisji CO2 w skali logarytmicznej
# fig, ax1 = plt.subplots(figsize=(12, 6))

# # Oś pierwsza (dla gas_price, coal_pscmi1_pln_per_gj, brent_price) w skali logarytmicznej
# ax1.plot(monthly_data["year_month"], monthly_data["gas_price"], label="Cena gazu (PLN/MWh)", color="blue", linewidth=2)
# ax1.plot(monthly_data["year_month"], monthly_data["co2_price"], label="Cena CO$_2$ (PLN/tCO$_2$)", color="red", linewidth=2)
# ax1.plot(monthly_data["year_month"], monthly_data["brent_price"], label="Cena ropy Brent (PLN/bar)", color="green", linewidth=2)
# ax1.set_xlabel("Rok", fontsize=12)
# ax1.set_ylabel("Cena (PLN/MWh, PLN/tCO$_2$, PLN/bar)", fontsize=12)
# ax1.set_yscale("log")  # Skala logarytmiczna dla osi Y
# ax1.legend(loc="upper left")
# ax1.grid(True, linestyle="--", alpha=0.7)

# # Oś druga (dla coal_pscmi1_pln_per_gj) w skali logarytmicznej
# ax2 = ax1.twinx()
# ax2.plot(monthly_data["year_month"], monthly_data["coal_pscmi1_pln_per_gj"], label="Cena węgla (PLN/GJ)", color="black", linewidth=2)
# ax2.set_ylabel("Cena węgla (PLN/GJ)", fontsize=12)
# ax2.set_yscale("log")  # Skala logarytmiczna dla osi Y
# ax2.legend(loc="upper right")

# # Ustawienie etykiet na osi X (co roku, tylko rok)
# ax1.xaxis.set_major_locator(mdates.YearLocator())  # Etykiety co roku
# ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # Format: tylko rok (np. 2016)
# plt.xticks(rotation=45, ha="right")

# plt.title("Ceny paliw kopalnych i emisji CO$_2$ (2016-2023) w skali logarytmicznej", fontsize=14)
# plt.tight_layout()
# plt.savefig(f"{output_dir}/fuel_prices_2016_2024.png", dpi=300, bbox_inches="tight")
# plt.close()

# # Utworzenie folderu losses w folderze plots, jeśli nie istnieje
# output_dir = "C:/mgr/EPF-Thesis/plots/losses"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Agregacja danych do średnich miesięcznych
# data["year_month"] = data["timestamp"].dt.to_period("M")
# monthly_data = data.groupby("year_month")[["power_loss", "Network_loss"]].mean().reset_index()
# monthly_data["year_month"] = monthly_data["year_month"].dt.to_timestamp()

# # Wykres liniowy dla strat mocy
# plt.figure(figsize=(12, 6))
# plt.plot(monthly_data["year_month"], monthly_data["power_loss"], label="Straty mocy stacji elektrycznych (MW)", color="red", linewidth=2)
# plt.plot(monthly_data["year_month"], monthly_data["Network_loss"], label="Utrata mocy w sieci (MW)", color="blue", linewidth=2)

# # Ustawienie etykiet na osi X (co roku, tylko rok)
# plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Etykiety co roku
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # Format: tylko rok (np. 2016)
# plt.xticks(rotation=45, ha="right")

# plt.xlabel("Rok", fontsize=12)
# plt.ylabel("Straty mocy (MW)", fontsize=12)
# plt.title("Straty mocy w wyniku awarii i w sieci (2016-2023)", fontsize=14)
# plt.legend(loc="upper left")
# plt.grid(True, linestyle="--", alpha=0.7)
# plt.tight_layout()
# plt.savefig(f"{output_dir}/power_losses.png", dpi=300, bbox_inches="tight")
# plt.close()

# # Filtrowanie danych dla okresu od marca 2024 do grudnia 2024
# data["date"] = data["timestamp"].dt.date
# start_date = pd.to_datetime("2024-03-01")
# end_date = pd.to_datetime("2024-12-31")
# filtered_data = data[(data["timestamp"] >= start_date) & (data["timestamp"] <= end_date)]

# # Agregacja danych do średnich dziennych
# daily_data = filtered_data.groupby("date")[["Network_loss"]].mean().reset_index()
# daily_data["date"] = pd.to_datetime(daily_data["date"])

# # Wykres liniowy dla network_loss w 2024 roku
# plt.figure(figsize=(12, 6))
# plt.plot(daily_data["date"], daily_data["Network_loss"], label="Utrata mocy w sieci (MW)", color="blue", linewidth=2)

# # Ustawienie etykiet na osi X (co miesiąc)
# plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Etykiety co miesiąc
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # Format: rok-miesiąc (np. 2024-03)
# plt.xticks(rotation=45, ha="right")

# plt.xlabel("Data", fontsize=12)
# plt.ylabel("Straty mocy w sieci (MW)", fontsize=12)
# plt.title("Straty mocy w sieci (marzec 2024 – grudzień 2024)", fontsize=14)
# plt.legend(loc="upper left")
# plt.grid(True, linestyle="--", alpha=0.7)
# plt.tight_layout()
# plt.savefig(f"{output_dir}/network_loss_2024.png", dpi=300, bbox_inches="tight")
# plt.close()

# # Wykres dla Load i fixing_i_volume w ujęciu miesięcznym (2016-2023)
# # Utworzenie folderu market w folderze plots, jeśli nie istnieje
# output_dir = "C:/mgr/EPF-Thesis/plots/market"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Agregacja danych do średnich miesięcznych
# data["year_month"] = data["timestamp"].dt.to_period("M")
# monthly_data = data.groupby("year_month")[["Load", "fixing_i_volume"]].mean().reset_index()
# monthly_data["year_month"] = monthly_data["year_month"].dt.to_timestamp()

# # Wykres liniowy dla Load i fixing_i_volume
# fig, ax1 = plt.subplots(figsize=(12, 6))

# # Oś pierwsza (dla Load)
# ax1.plot(monthly_data["year_month"], monthly_data["Load"], label="Zapotrzebowanie (MW)", color="blue", linewidth=2)
# ax1.set_xlabel("Rok", fontsize=12)
# ax1.set_ylabel("Zapotrzebowanie (MW)", fontsize=12, color="blue")
# ax1.tick_params(axis="y", labelcolor="blue")
# ax1.grid(True, linestyle="--", alpha=0.7)

# # Oś druga (dla fixing_i_volume)
# ax2 = ax1.twinx()
# ax2.plot(monthly_data["year_month"], monthly_data["fixing_i_volume"], label="Wolumen sprzedaży na RDN (MWh)", color="red", linewidth=2)
# ax2.set_ylabel("Wolumen sprzedaży (MWh)", fontsize=12, color="red")
# ax2.tick_params(axis="y", labelcolor="red")

# # Ustawienie etykiet na osi X (co roku)
# ax1.xaxis.set_major_locator(mdates.YearLocator())  # Etykiety co roku
# ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # Format: tylko rok (np. 2016)
# plt.xticks(rotation=45, ha="right")

# # Tytuł i legenda
# plt.title("Średnie miesięczne zapotrzebowanie i wolumen sprzedaży na RDN (2016-2023)", fontsize=14)
# fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
# plt.tight_layout()
# plt.savefig(f"{output_dir}/load_vs_volume_2016_2024.png", dpi=300, bbox_inches="tight")
# plt.close()

# # Grupowanie danych według roku i obliczanie średniej wartości pln_usd
# annual_pln_usd = data.groupby("year")["pln_usd"].mean().reset_index()
# annual_pln_eur = data.groupby("year")["pln_eur"].mean().reset_index()

# # Wyświetlenie wyników na konsoli
# print("Średnioroczny kurs PLN/USD w latach 2016-2023:")
# for index, row in annual_pln_usd.iterrows():
#     print(f"Rok: {int(row['year'])}, Średni kurs PLN/USD: {row['pln_usd']:.2f}")

# for index, row in annual_pln_eur.iterrows():
#     print(f"Rok: {int(row['year'])}, Średni kurs PLN/EUR: {row['pln_eur']:.2f}")

# # STATYSTYKI RYNKOWE TEKSTOWE 
# # Lista zmiennych cenowych
# price_columns = ['fixing_i_price', 'se_price', 'sk_price', 'cz_price', 'lt_price']

# # Obliczamy statystyki dla całego okresu
# stats_dict = {}
# for column in price_columns:
#     stats = {
#         'Średnia': df[column].mean(),
#         'Odchylenie std.': df[column].std(),
#         'Minimum': df[column].min(),
#         '25% (Q1)': df[column].quantile(0.25),
#         'Mediana': df[column].median(),
#         '75% (Q3)': df[column].quantile(0.75),
#         'Maksimum': df[column].max()
#     }
#     stats_dict[column] = stats

# # Tworzymy DataFrame z statystykami
# stats_df = pd.DataFrame(stats_dict).round(2)

# # Wyświetlamy statystyki w konsoli
# print("\nStatystyki cen rynków (PLN/MWh) dla okresu 2016-2023:")
# print(stats_df)

# # Filtruj dane dla lt_price > 10000 i wyświetl timestamp
# high_lt_price_timestamps = df[df["lt_price"] > 5000]["timestamp"]
# print("Timestamps dla lt_price > 5000:")
# print(high_lt_price_timestamps)

# # Oblicz procent przypadków, gdzie Network_loss wynosi 0
# network_loss_zero_percentage = (df["Network_loss"] == 0).mean() * 100
# print(f"Procent przypadków, gdzie Network_loss wynosi 0: {network_loss_zero_percentage:.2f}%")

# # WYKRES EMISYJNE I NIE-EMISYJNE ŹRÓDŁA ENERGII
# output_dir = "C:/mgr/EPF-Thesis/plots/energy"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Filtrowanie danych dla roku 2023
# data_2023 = df[df["timestamp"].dt.year == 2023]

# # Obliczanie emisji i nie-emisji w wartościach liczbowych
# data_2023["non_emissive_sources"] = data_2023["solar"] + data_2023["wind_onshore"] + data_2023["biomass"]
# data_2023["emissive_sources"] = (
#     data_2023["hard_coal"] + data_2023["coal-derived"] + data_2023["lignite"] +
#     data_2023["gas"] + data_2023["oil"]
# )

# # Wykres obszarowy emisji vs nie-emisji
# plt.figure(figsize=(16, 9))  # Większa rozdzielczość
# plt.stackplot(
#     data_2023["timestamp"],
#     data_2023["non_emissive_sources"],
#     data_2023["emissive_sources"],
#     labels=["Nie-emisyjne źródła (MW)", "Emisyjne źródła (MW)"],
#     colors=["green", "red"]
# )

# # Dodanie etykiet i tytułu
# plt.title("Produkcja energii z emisyjnych i nie-emisyjnych źródeł w 2023 roku", fontsize=16)
# plt.xlabel("Data", fontsize=14)
# plt.ylabel("Produkcja energii (MW)", fontsize=14)
# plt.legend(loc="upper left", fontsize=12)
# plt.grid(True, linestyle="--", alpha=0.7)

# # Formatowanie osi X dla dat
# plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Etykiety co miesiąc
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # Format: rok-miesiąc
# plt.xticks(rotation=45, ha="right")

# # Dopasowanie układu i zapis wykresu
# plt.tight_layout()
# plt.savefig(f"{output_dir}/emission_vs_non_emission_2023_high_res.png", dpi=300)
# plt.close()

# # Oblicz średnie wartości dla non_emissive_sources_percentage
# average_non_emissive_2023 = data_2023["non_emissive_sources_percentage"].mean()
# average_non_emissive_all = df["non_emissive_sources_percentage"].mean()

# # Wyświetl wyniki
# print(f"Średnia wartość non_emissive_sources_percentage dla roku 2023: {average_non_emissive_2023:.2f}%")
# print(f"Średnia wartość non_emissive_sources_percentage dla całego okresu: {average_non_emissive_all:.2f}%")

# # RB - WYKRES
# # Tworzenie folderu na wykresy, jeśli nie istnieje

# # Tworzenie folderu na wykresy, jeśli nie istnieje
# output_dir = "C:/mgr/EPF-Thesis/plots/market"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Wykres porównujący rb_price oraz fixing_i_price
# plt.figure(figsize=(12, 6))
# plt.plot(df["timestamp"], df["RB_price"], label="RB Price (PLN/MWh)", color="blue", linewidth=2)
# plt.plot(df["timestamp"], df["fixing_i_price"], label="Fixing I Price (PLN/MWh)", color="red", linewidth=2)

# # Dodanie etykiet i tytułu
# plt.title("Porównanie cen RB i RDN (2016-2023)", fontsize=14)
# plt.xlabel("Czas", fontsize=12)
# plt.ylabel("Cena (PLN/MWh)", fontsize=12)
# plt.legend(loc="upper left")
# plt.grid(True, linestyle="--", alpha=0.7)

# # Formatowanie osi X dla dat
# plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Etykiety co roku
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # Format: tylko rok
# plt.xticks(rotation=45, ha="right")

# # Dopasowanie układu i zapis wykresu
# plt.tight_layout()
# plt.savefig(f"{output_dir}/rb_vs_fixing_i_price.png", dpi=300)
# plt.close()

# # Oblicz średnie odchylenie RB_price od fixing_i_price
# df["price_deviation"] = df["RB_price"] - df["fixing_i_price"]
# average_deviation = df["price_deviation"].mean()

# # Wyświetl wynik
# print(f"Średnie odchylenie RB_price od fixing_i_price: {average_deviation:.2f} PLN/MWh")

# # Wykres średniego odchylenia RB_price od fixing_i_price w ujęciu miesięcznym
# # Agregacja danych do średnich miesięcznych
# df["year_month"] = df["timestamp"].dt.to_period("M")
# monthly_deviation = df.groupby("year_month")["price_deviation"].mean().reset_index()
# monthly_deviation["year_month"] = monthly_deviation["year_month"].dt.to_timestamp()

# # Tworzenie wykresu
# plt.figure(figsize=(12, 6))
# plt.plot(monthly_deviation["year_month"], monthly_deviation["price_deviation"], color="blue", linewidth=2, label="Średnie odchylenie")

# # Dodanie linii zerowej
# plt.axhline(0, color="black", linestyle="--", alpha=0.7)

# # Dodanie etykiet i tytułu
# plt.title("Średnie miesięczne odchylenie RB_price od fixing_i_price (2016-2023)", fontsize=14)
# plt.xlabel("Rok", fontsize=12)
# plt.ylabel("Średnie odchylenie miesięczne (PLN/MWh)", fontsize=12)
# plt.legend(loc="upper left")
# plt.grid(True, linestyle="--", alpha=0.7)

# # Formatowanie osi X
# plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Etykiety co roku
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # Format: tylko rok
# plt.xticks(rotation=45, ha="right")

# # Dopasowanie układu i zapis wykresu
# plt.tight_layout()
# output_dir = "C:/mgr/EPF-Thesis/plots/market"
# plt.savefig(f"{output_dir}/average_price_deviation.png", dpi=300)
# plt.close()
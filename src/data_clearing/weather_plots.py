import pandas as pd
import matplotlib.pyplot as plt
import os

# Utworzenie folderu weather w folderze plots, jeśli nie istnieje
output_dir = "C:/mgr/EPF-Thesis/plots/weather"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Wczytanie danych
data = pd.read_csv("C:/mgr/EPF-Thesis/data/database.csv", parse_dates=["timestamp"])

# 1. Szeregi czasowe dla całego okresu (2016–2024) w ujęciu miesięcznym
# Agregacja danych do średnich miesięcznych
data["year_month"] = data["timestamp"].dt.to_period("M")
monthly_data = data.groupby("year_month").mean(numeric_only=True).reset_index()
monthly_data["year_month"] = monthly_data["year_month"].astype(str)

# Temperatura (cały okres)
plt.figure(figsize=(12, 6))
plt.plot(monthly_data["year_month"], monthly_data["temp_waw"], label="Warszawa", color="#3498db")
plt.plot(monthly_data["year_month"], monthly_data["temp_ksz"], label="Koszalin", color="#e74c3c")
plt.plot(monthly_data["year_month"], monthly_data["temp_krk"], label="Kraków", color="#2ecc71")
plt.plot(monthly_data["year_month"], monthly_data["temp_bab"], label="Babimost", color="#f1c40f")
plt.title("Średnia miesięczna temperatura w różnych lokalizacjach (2016–2024)", fontsize=14)
plt.xlabel("Rok-Miesiąc", fontsize=12)
plt.ylabel("Temperatura (°C)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(rotation=45, ha="right")
plt.gca().set_xticks(range(0, len(monthly_data), 6))  # Pokazuj co 6. miesiąc dla czytelności
plt.tight_layout()
plt.savefig(f"{output_dir}/temp_time_series_full.png", dpi=300)
plt.close()

# Prędkość wiatru (cały okres)
plt.figure(figsize=(12, 6))
plt.plot(monthly_data["year_month"], monthly_data["wind_speed_waw"], label="Warszawa", color="#3498db")
plt.plot(monthly_data["year_month"], monthly_data["wind_speed_ksz"], label="Koszalin", color="#e74c3c")
plt.plot(monthly_data["year_month"], monthly_data["wind_speed_krk"], label="Kraków", color="#2ecc71")
plt.plot(monthly_data["year_month"], monthly_data["wind_speed_bab"], label="Babimost", color="#f1c40f")
plt.title("Średnia miesięczna prędkość wiatru w różnych lokalizacjach (2016–2024)", fontsize=14)
plt.xlabel("Rok-Miesiąc", fontsize=12)
plt.ylabel("Prędkość wiatru (m/s)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(rotation=45, ha="right")
plt.gca().set_xticks(range(0, len(monthly_data), 6))
plt.tight_layout()
plt.savefig(f"{output_dir}/wind_speed_time_series_full.png", dpi=300)
plt.close()

# Zachmurzenie (cały okres)
plt.figure(figsize=(12, 6))
plt.plot(monthly_data["year_month"], monthly_data["cloud_cover_waw"], label="Warszawa", color="#3498db")
plt.plot(monthly_data["year_month"], monthly_data["cloud_cover_ksz"], label="Koszalin", color="#e74c3c")
plt.plot(monthly_data["year_month"], monthly_data["cloud_cover_krk"], label="Kraków", color="#2ecc71")
plt.plot(monthly_data["year_month"], monthly_data["cloud_cover_bab"], label="Babimost", color="#f1c40f")
plt.title("Średnie miesięczne zachmurzenie w różnych lokalizacjach (2016–2024)", fontsize=14)
plt.xlabel("Rok-Miesiąc", fontsize=12)
plt.ylabel("Zachmurzenie (Oktanty)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(rotation=45, ha="right")
plt.gca().set_xticks(range(0, len(monthly_data), 6))
plt.tight_layout()
plt.savefig(f"{output_dir}/cloud_cover_time_series_full.png", dpi=300)
plt.close()

# Promieniowanie słoneczne (cały okres)
plt.figure(figsize=(12, 6))
plt.plot(monthly_data["year_month"], monthly_data["solar_radiation_waw"], label="Warszawa", color="#3498db")
plt.plot(monthly_data["year_month"], monthly_data["solar_radiation_ksz"], label="Koszalin", color="#e74c3c")
plt.plot(monthly_data["year_month"], monthly_data["solar_radiation_krk"], label="Kraków", color="#2ecc71")
plt.plot(monthly_data["year_month"], monthly_data["solar_radiation_bab"], label="Babimost", color="#f1c40f")
plt.title("Średnie miesięczne promieniowanie słoneczne w różnych lokalizacjach (2016–2024)", fontsize=14)
plt.xlabel("Rok-Miesiąc", fontsize=12)
plt.ylabel("Promieniowanie słoneczne (W/m²)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(rotation=45, ha="right")
plt.gca().set_xticks(range(0, len(monthly_data), 6))
plt.tight_layout()
plt.savefig(f"{output_dir}/solar_radiation_time_series_full.png", dpi=300)
plt.close()

# 2. Szeregi czasowe dla roku 2022 w ujęciu dziennym
# Filtrowanie danych dla roku 2022
data_2022 = data[data["timestamp"].dt.year == 2022]
data_2022["day"] = data_2022["timestamp"].dt.date
daily_data_2022 = data_2022.groupby("day").mean(numeric_only=True).reset_index()
daily_data_2022["day"] = pd.to_datetime(daily_data_2022["day"])

# Temperatura (2022)
plt.figure(figsize=(12, 6))
plt.plot(daily_data_2022["day"], daily_data_2022["temp_waw"], label="Warszawa", color="#3498db")
plt.plot(daily_data_2022["day"], daily_data_2022["temp_ksz"], label="Koszalin", color="#e74c3c")
plt.plot(daily_data_2022["day"], daily_data_2022["temp_krk"], label="Kraków", color="#2ecc71")
plt.plot(daily_data_2022["day"], daily_data_2022["temp_bab"], label="Babimost", color="#f1c40f")
plt.title("Średnia dzienna temperatura w różnych lokalizacjach w 2022 roku", fontsize=14)
plt.xlabel("Data", fontsize=12)
plt.ylabel("Temperatura (°C)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(rotation=45, ha="right")
plt.gca().set_xticks(daily_data_2022["day"][::30])  # Pokazuj co 30. dzień dla czytelności
plt.tight_layout()
plt.savefig(f"{output_dir}/temp_time_series_2022.png", dpi=300)
plt.close()

# Prędkość wiatru (2022)
plt.figure(figsize=(12, 6))
plt.plot(daily_data_2022["day"], daily_data_2022["wind_speed_waw"], label="Warszawa", color="#3498db")
plt.plot(daily_data_2022["day"], daily_data_2022["wind_speed_ksz"], label="Koszalin", color="#e74c3c")
plt.plot(daily_data_2022["day"], daily_data_2022["wind_speed_krk"], label="Kraków", color="#2ecc71")
plt.plot(daily_data_2022["day"], daily_data_2022["wind_speed_bab"], label="Babimost", color="#f1c40f")
plt.title("Średnia dzienna prędkość wiatru w różnych lokalizacjach w 2022 roku", fontsize=14)
plt.xlabel("Data", fontsize=12)
plt.ylabel("Prędkość wiatru (m/s)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(rotation=45, ha="right")
plt.gca().set_xticks(daily_data_2022["day"][::30])
plt.tight_layout()
plt.savefig(f"{output_dir}/wind_speed_time_series_2022.png", dpi=300)
plt.close()

# Zachmurzenie (2022)
plt.figure(figsize=(12, 6))
plt.plot(daily_data_2022["day"], daily_data_2022["cloud_cover_waw"], label="Warszawa", color="#3498db")
plt.plot(daily_data_2022["day"], daily_data_2022["cloud_cover_ksz"], label="Koszalin", color="#e74c3c")
plt.plot(daily_data_2022["day"], daily_data_2022["cloud_cover_krk"], label="Kraków", color="#2ecc71")
plt.plot(daily_data_2022["day"], daily_data_2022["cloud_cover_bab"], label="Babimost", color="#f1c40f")
plt.title("Średnie dzienne zachmurzenie w różnych lokalizacjach w 2022 roku", fontsize=14)
plt.xlabel("Data", fontsize=12)
plt.ylabel("Zachmurzenie (Oktanty)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(rotation=45, ha="right")
plt.gca().set_xticks(daily_data_2022["day"][::30])
plt.tight_layout()
plt.savefig(f"{output_dir}/cloud_cover_time_series_2022.png", dpi=300)
plt.close()

# Promieniowanie słoneczne (2022)
plt.figure(figsize=(12, 6))
plt.plot(daily_data_2022["day"], daily_data_2022["solar_radiation_waw"], label="Warszawa", color="#3498db")
plt.plot(daily_data_2022["day"], daily_data_2022["solar_radiation_ksz"], label="Koszalin", color="#e74c3c")
plt.plot(daily_data_2022["day"], daily_data_2022["solar_radiation_krk"], label="Kraków", color="#2ecc71")
plt.plot(daily_data_2022["day"], daily_data_2022["solar_radiation_bab"], label="Babimost", color="#f1c40f")
plt.title("Średnie dzienne promieniowanie słoneczne w różnych lokalizacjach w 2022 roku", fontsize=14)
plt.xlabel("Data", fontsize=12)
plt.ylabel("Promieniowanie słoneczne (W/m²)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(rotation=45, ha="right")
plt.gca().set_xticks(daily_data_2022["day"][::30])
plt.tight_layout()
plt.savefig(f"{output_dir}/solar_radiation_time_series_2022.png", dpi=300)
plt.close()
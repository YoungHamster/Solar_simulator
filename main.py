import math
import numpy as np
import sys
import matplotlib.pyplot as plt

from heatmap import heatmap
from heatmap import annotate_heatmap

# выработка солнечного массива номинальной мощностью 1 квт от сайта https://pvwatts.nrel.gov/pvwatts.php без учёта сезонных наклонов панелей(с наклонами выработка будет выше)
energy_generated_per_month = [32.4, 63, 109.9, 116.5, 153.6, 155.8, 158.7, 153.4, 93.4, 64, 18.4, 14.2]

months = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь",
          "декабрь"]

kW_price_rub = 18100
kWh_battery_price_rub = 6753
bms_price_rub = 12819
max_6kW_inverter_price_rub = 35707
max_11kW_inverter_price_rub = 96328

# profit/h исчисляется в долларах
miners = [{'name': 'antminer s21', 'price': 192000, 'power': 3.5, 'hashrate': 200},
          #{'name':'antminer s19k pro', 'price':80000, 'power':2.8, 'profit/h':0.2979},
          #{'name':'antminer s19 100th', 'price':31800, 'power':3.25, 'profit/h':0.2487},
          #{'name':'antminer z15 pro', 'price':190650, 'power':2.7, 'profit/h':0.4333},
          #{'name':'antminer s19j pro 92th', 'price':42454, 'power':2.5, 'profit/h':0.22885},
          {'name': 'antminer s19xp', 'price': 62000, 'power': 2.881, 'hashrate': 141},
          {'name': 'innosilicon t2thf+', 'price': 8000, 'power': 2.2, 'hashrate': 33}
          ]

dollar_price = 78.11
usd_perhash_per24h = 0.0322

solar_hours_avg_day = [2.5, 2.9, 4.5, 7.2, 9.8, 10.8, 11.2, 11.2, 8.5, 5.8, 4.2, 2.6]
solar_days = [10, 9, 14, 20, 25, 26, 28, 29, 25, 20, 17, 11]
days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
solar_day_length_avg = [8.5, 10, 12, 14, 15.66, 16.66, 16, 14.66, 12.66, 10.66, 9, 8]


def hourly_energy_approximation(
        days_in_month,  # Число дней в каждом месяце (12)
        sunny_days_in_month,  # Число солнечных дней (12)
        avg_sun_hours_per_day,  # Среднее солнечных часов в день (12)
        avg_energy_production,  # Средняя выработка энергии (12)
        day_length,  # Длина светового дня (12)
        cloud_efficiency=0.5,  # Эффективность в облака (0-1)
        seed=None  # Seed для воспроизводимости
):
    """Аппроксимирует почасовую выработку солнечной энергии с учетом:
    - Случайного распределения солнечных дней
    - Колоколообразного профиля выработки
    - Облачных дней с частичной эффективностью
    - Сезонных изменений светового дня

    Возвращает массив выработки энергии по часам для всего года"""

    # Инициализация генератора случайных чисел
    rng = np.random.default_rng(seed)

    # Подготовка данных
    total_days = sum(days_in_month)
    result = np.zeros(total_days * 24)
    day_start_index = 0

    # Параметры колоколообразного распределения
    PEAK_HOUR = 12.0  # Пик выработки в полдень
    SIGMA_FACTOR = 5  # Коэффициент ширины колокола для солнечного дня

    for month in range(12):
        n_days = days_in_month[month]
        n_sunny = sunny_days_in_month[month]
        monthly_energy = avg_energy_production[month]
        daylight = day_length[month]

        # 1. Случайное распределение солнечных часов в месяце
        sunny_hour_flags = np.zeros(n_days * 24, dtype=bool)
        sunny_indices = rng.choice(n_days * 24, size=int(n_sunny * 24), replace=False)
        sunny_hour_flags[sunny_indices] = True

        # 3. Расчет профиля дня (колоколообразная кривая)
        sun_energy_profile = np.zeros(24)

        # Параметры нормального распределения
        sigma = daylight / SIGMA_FACTOR
        total_weight = 0
        hour_weights = []

        normal_distribution_factor = 1 / math.sqrt(2 * math.pi * sigma ** 2)
        for hour in range(24):
            hour_center = hour + 0.5
            # Рассчитываем вес только в пределах светового дня
            if abs(hour_center - PEAK_HOUR) <= daylight / 2:
                weight_sunny = math.exp(-(hour_center - PEAK_HOUR) ** 2 / (2 * sigma ** 2))
                weight_sunny = weight_sunny * normal_distribution_factor
            else:
                weight_sunny = 0

            hour_weights.append(weight_sunny)
            total_weight += weight_sunny

        # Нормализуем и масштабируем веса
        if total_weight > 0:
            sun_energy_profile = np.array(hour_weights) / total_weight

        # 4. Расчет энергии для солнечных и облачных дней
        # Энергия на один час
        month_multiplicator = monthly_energy / energy_generated_per_month[0]

        # Для каждого дня в месяце
        for day in range(n_days):
            day_profile = sun_energy_profile * month_multiplicator
            # Добавляем профиль дня к результату
            start_idx = day_start_index + day * 24
            end_idx = start_idx + 24
            result[start_idx:end_idx] = day_profile

        day_start_index += n_days * 24

    return result


def calc_profit_for_year(energy_gen_by_hour, solar_kw, battery_kwh, miner):
    battery_efficiency = 0.95

    effectively_storable_energy = 0
    generated_rub = []
    profit_per_hour = miner['hashrate'] * usd_perhash_per24h * dollar_price / 24
    for energy in energy_gen_by_hour:
        # рассчет энергии идёт для солнечных панелей номиналом 1квт,
        # если у нас больше/меньше то необходимо включать множитель
        generated_energy = energy * solar_kw
        if generated_energy >= miner['power']:
            generated_rub.append(profit_per_hour)

            effectively_storable_energy += (generated_energy - miner['power']) * battery_efficiency
            if effectively_storable_energy > battery_kwh: effectively_storable_energy = battery_kwh
        else:
            if effectively_storable_energy + generated_energy > miner['power']:
                generated_rub.append(profit_per_hour)
                effectively_storable_energy -= miner['power'] - generated_energy
            else:
                time_mining = (effectively_storable_energy + generated_energy) / miner['power']
                generated_rub.append(profit_per_hour * time_mining)
                effectively_storable_energy = 0

    return sum(generated_rub)

def find_best_payback(minkW, maxkW, minkWh, maxkWh, steps):
    min_payback = 100
    best_miner = 0
    max_roi_i = 0
    max_roi_j = 0
    energy_profile = hourly_energy_approximation(
        days,
        solar_days,
        solar_hours_avg_day,
        energy_generated_per_month,
        solar_day_length_avg
    )
    np.set_printoptions(threshold=sys.maxsize)
    print(energy_profile)
    kW_step = (maxkW - minkW) / steps
    kWh_step = (maxkWh - minkWh) / steps

    for miner in miners:
        payback_years = [[] for i in range(steps)]
        for i in range(steps):
            for j in range(steps):
                solar_kW = i * kW_step + minkW
                battery_kWh = j * kWh_step + minkWh
                price = miners[0]['price'] + kW_price_rub * solar_kW + kWh_battery_price_rub * battery_kWh
                profit = calc_profit_for_year(energy_profile, i, j, miner)
                if profit != 0:
                    payback_years[i].append(price / profit)
                else:
                    payback_years[i].append(100)
                    print(f"Profit = 0 for miner {miner['name']}, kW: {solar_kW}, kWh: {battery_kWh}")

        for i in range(steps):
            for j in range(steps):
                if payback_years[i][j] < min_payback:
                    max_roi_i = i
                    max_roi_j = j
                    best_miner = miner

    print(f"Best miner: {best_miner['name']}, payback years: {min_payback}, solar kW: {minkW + max_roi_i * kW_step}, battery kWh: {minkWh + max_roi_j * kWh_step}")

find_best_payback(5, 30, 2, 30, 10)
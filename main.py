import math
import numpy as np
import matplotlib.pyplot as plt

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

        print(f"month_multiplicator = {round(month_multiplicator, 2)}, sun_hours = {round(n_sunny*daylight, 0)}"
              f", cloud_hours = {round((n_days - n_sunny) * daylight, 0)},"
              f" effective hours = {round(n_sunny * daylight + (n_days - n_sunny) * daylight * cloud_efficiency, 0)}"
              f" energy = {monthly_energy}")

        # Для каждого дня в месяце
        for day in range(n_days):
            day_profile = sun_energy_profile * month_multiplicator
            # Добавляем профиль дня к результату
            start_idx = day_start_index + day * 24
            end_idx = start_idx + 24
            result[start_idx:end_idx] = day_profile

        day_start_index += n_days * 24

    return result


# Вызов функции
result = hourly_energy_approximation(
    days,
    solar_days,
    solar_hours_avg_day,
    energy_generated_per_month,
    solar_day_length_avg
)

# Визуализация случайной недели
day_index = 180  # Произвольный день
start = day_index * 24
end = start + 24

plt.figure(figsize=(20, 6))
plt.plot(result[start:end])
plt.title(f"Почасовая выработка энергии (День {day_index})")
plt.xlabel("Час дня")
plt.ylabel("Выработка энергии")
plt.grid(True)
plt.show()

month_production = [sum(result[sum(days[:i])*24: sum(days[:i+1])*24]) for i in range(12)]

for i in range(12):
    print(f"За мессяц {i} выработано {month_production[i]} квтч, относительно июля это {month_production[7] / month_production[i]}, ожидаемое соотношение: {energy_generated_per_month[7] / energy_generated_per_month[i]}")


def calc_profit_metrics(energy_gen_by_month, main_miner, second_miner):
    solar_hours_avg_day = [2.5, 2.9, 4.5, 7.2, 9.8, 10.8, 11.2, 11.2, 8.5, 5.8, 4.2, 2.6]
    solar_days = [10, 9, 14, 20, 25, 26, 28, 29, 25, 20, 17, 11]
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    solar_day_length_avg = [8.5, 10, 12, 14, 15.66, 16.66, 16, 14.66, 12.66, 10.66, 9, 8]
    peak_hour_share = 0.384
    peak_hour_power_share = 0.588

    profit_with_battery = []
    profit_without_battery = []
    profit_with_second_miner = []

    for i in range(12):
        solar_hours = solar_hours_avg_day[i] * days[i]
        sunny_hours_energy_share = solar_hours / (solar_day_length_avg[i] * days[i] * 0.8)
        if sunny_hours_energy_share > 1:
            sunny_hours_energy_share = 1
        kWh_sunny = energy_gen_by_month[i] * sunny_hours_energy_share

        energy_per_peak_hour = (kWh_sunny * peak_hour_power_share) / (solar_hours * peak_hour_share)
        excess_kWh = 0
        if energy_per_peak_hour > main_miner['power']:
            excess_kWh = (energy_per_peak_hour - main_miner['power']) * (solar_hours * peak_hour_share)

        energy_per_off_peak_hour = (kWh_sunny * (1 - peak_hour_power_share)) / (solar_hours * (1 - peak_hour_share))
        if energy_per_off_peak_hour > main_miner['power']:
            excess_kWh += (energy_per_off_peak_hour - main_miner['power']) * (solar_hours * (1 - peak_hour_share))

        mining_hours_no_battery = (energy_gen_by_month[i] - excess_kWh) / main_miner['power']
        mining_hours_battery = energy_gen_by_month[i] / main_miner['power']
        if mining_hours_battery > days[i] * 24:
            mining_hours_battery = days[i] * 24
        profit_with_battery.append(mining_hours_battery * main_miner['profit/h'] * dollar_price)
        profit_without_battery.append(mining_hours_no_battery * main_miner['profit/h'] * dollar_price)

    return profit_with_battery, profit_without_battery, profit_with_second_miner


def calc_roi_for_miner(miner, minkW, maxkW, steps):
    battery_system_price = []
    battery_roi = []
    no_battery_system_price = []
    no_battery_roi = []
    battery_price = []
    step = (maxkW - minkW) / steps
    for i in range(steps):
        solar_kW = minkW + i * step
        (profit_with_without_battery_per_month, energy_per_peak_hour_per_month, sunny_hour_energy_share_per_month,
         excess_kWh_per_sunny_day_per_month) = calc_profit_metrics([x * solar_kW for x in energy_generated_per_month],
                                                                   miner['power'], miner['profit/h'])

        if solar_kW <= 6:
            system_price = solar_kW * kW_price_rub + miner['price'] + max_6kW_inverter_price_rub
        else:
            system_price = solar_kW * kW_price_rub + miner['price'] + max_6kW_inverter_price_rub * 2
        profit_with_battery_per_month = [x[0] for x in profit_with_without_battery_per_month]
        profit_without_battery_per_month = [x[1] for x in profit_with_without_battery_per_month]

        depreciation = miner['price'] * 0.1
        profit_per_year_no_battery = sum(profit_without_battery_per_month) - depreciation
        no_battery_system_price.append(system_price)
        roi = profit_per_year_no_battery / system_price * 100
        no_battery_roi.append(roi)

        profit_per_year_battery = sum(profit_with_battery_per_month) - depreciation
        bpr = max(excess_kWh_per_sunny_day_per_month) * kWh_battery_price_rub
        battery_price.append(bpr)
        system_price_battery = bpr + system_price
        roi_battery = profit_per_year_battery / system_price_battery * 100
        battery_system_price.append(system_price_battery)
        battery_roi.append(roi_battery)

    max_roi = 0
    max_index = 0
    for i in range(len(battery_roi)):
        if battery_roi[i] > max_roi:
            max_roi = battery_roi[i]
            max_index = i
    solar_price = (minkW + step * max_index) * kW_price_rub
    print(miner['name'], 'Max ROI:', max_roi, 'system price:', battery_system_price[max_index],
          'battery price:', battery_price[max_index], 'battery size:', battery_price[max_index] / kWh_battery_price_rub,
          'solar price:', solar_price, 'solar size:', solar_price / kW_price_rub,
          'miner price:', miner['price'],
          'inverter price:', battery_system_price[max_index] - miner['price'] - solar_price - battery_price[max_index])

    return no_battery_system_price, no_battery_roi, battery_system_price, battery_roi


def graph_rois_by_miner():
    minkW = 4
    maxkW = 16
    steps = 60
    for miner in miners:
        no_battery_price, no_battery_roi, battery_price, battery_roi = calc_roi_for_miner(miner, minkW, maxkW, steps)
        miner['nbp'] = no_battery_price
        miner['nbr'] = no_battery_roi
        miner['bp'] = battery_price
        miner['br'] = battery_roi

    fig = plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    for miner in miners:
        plt.plot([minkW + i * ((maxkW - minkW) / steps) for i in range(steps)], miner['bp'],
                 label=miner['name'] + '+batt')
        plt.plot([minkW + i * ((maxkW - minkW) / steps) for i in range(steps)], miner['nbp'], label=miner['name'])
    plt.xlabel('solar kW')
    plt.title('Стоимость системы')

    plt.subplot(2, 1, 2)
    for miner in miners:
        plt.plot([minkW + i * ((maxkW - minkW) / steps) for i in range(steps)], miner['br'],
                 label=miner['name'] + '+batt')
        plt.plot([minkW + i * ((maxkW - minkW) / steps) for i in range(steps)], miner['nbr'], label=miner['name'])
    plt.xlabel('solar kW')
    plt.title('ROI')

    plt.legend()
    plt.show()


def show_calc_profit_metrics(energy_gen_by_month, miner_power, miner_profit_per_h_usd):
    (profit_with_without_battery_per_month, energy_per_peak_hour_per_month, sunny_hour_energy_share_per_month,
     excess_kWh_per_sunny_day_per_month) = calc_profit_metrics(energy_gen_by_month,
                                                               miner_power, miner_profit_per_h_usd)

    fig = plt.figure(figsize=(12, 10))
    # прибыль с аккумом и без по месяцам
    plt.subplot(2, 2, 1)
    plt.plot(months, profit_with_without_battery_per_month)
    plt.ylabel('Рубли')
    plt.title('Прибыль за месяц с аккумуляторами и без')
    plt.grid(axis='y')

    # производство энергии по месяцам
    plt.subplot(2, 2, 2)
    plt.plot(months, energy_generated_per_month)
    plt.ylabel('kWh')
    plt.title('Энергия за месяц')
    plt.grid(axis='y')

    # прозиводство энергии в пиковые часы по месяцам
    plt.subplot(2, 2, 3)
    plt.plot(months, energy_per_peak_hour_per_month)
    plt.ylabel('kW')
    plt.title('Производство энергии в пиковые часы светового дня')
    plt.grid(axis='y')

    # доля солнечных часов в производстве энергии
    plt.subplot(2, 2, 4)
    plt.plot(months, sunny_hour_energy_share_per_month)
    plt.ylabel('%')
    plt.title('Доля энергии от солнечных часов в световом дне')
    plt.grid(axis='y')

    plt.show()

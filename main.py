import time

import matplotlib.pyplot as plt

energy_generated_per_month = [32.4, 63, 109.9, 116.5, 153.6, 155.8, 158.7, 153.4, 93.4, 64, 18.4, 14.2]

months = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь",
          "декабрь"]

kW_price_rub = 18900
kWh_battery_price_rub = 7552
max_6kW_inverter_price_rub = 35707
max_11kW_inverter_price_rub = 96328

miners = [{'name':'antminer s21', 'price': 192000, 'power':3.5, 'profit/h':0.4975},
          #{'name':'antminer s19k pro', 'price':80000, 'power':2.8, 'profit/h':0.2979},
          #{'name':'antminer s19 100th', 'price':31800, 'power':3.25, 'profit/h':0.2487},
          #{'name':'antminer z15 pro', 'price':190650, 'power':2.7, 'profit/h':0.4333},
          #{'name':'antminer s19j pro 92th', 'price':42454, 'power':2.5, 'profit/h':0.22885},
          {'name':'antminer s19xp', 'price':103955, 'power':2.881, 'profit/h':0.3333}
          ]

dollar_price = 78.11


def calc_profit_metrics(energy_gen_by_month, miner_power, miner_profit_per_h_usd):
    solar_hours_avg_day = [2.5, 2.9, 4.5, 7.2, 9.8, 10.8, 11.2, 11.2, 8.5, 5.8, 4.2, 2.6]
    solar_days = [10, 9, 14, 20, 25, 26, 28, 29, 25, 20, 17, 11]
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    solar_day_length_avg = [8.5, 10, 12, 14, 15.66, 16.66, 16, 14.66, 12.66, 10.66, 9, 8]
    peak_hour_share = 0.384
    peak_hour_power_share = 0.588

    profit_with_battery = []
    profit_without_battery = []
    profit_with_second_miner = []
    profit_with_without_battery = []

    for i in range(12):
        solar_hours = solar_hours_avg_day[i] * days[i]
        sunny_hours_energy_share = solar_hours / (solar_day_length_avg[i] * days[i] * 0.8)
        if sunny_hours_energy_share > 1:
            sunny_hours_energy_share = 1
        kWh_sunny = energy_gen_by_month[i] * sunny_hours_energy_share

        energy_per_peak_hour = (kWh_sunny * peak_hour_power_share) / (solar_hours * peak_hour_share)
        excess_kWh = 0
        if energy_per_peak_hour > miner_power:
            excess_kWh = (energy_per_peak_hour - miner_power) * (solar_hours * peak_hour_share)

        energy_per_off_peak_hour = (kWh_sunny * (1 - peak_hour_power_share)) / (solar_hours * (1 - peak_hour_share))
        if energy_per_off_peak_hour > miner_power:
            excess_kWh += (energy_per_off_peak_hour - miner_power) * (solar_hours * (1 - peak_hour_share))

        mining_hours_no_battery = (energy_gen_by_month[i] - excess_kWh) / miner_power
        mining_hours_battery = energy_gen_by_month[i] / miner_power
        if mining_hours_battery > days[i] * 24:
            mining_hours_battery = days[i] * 24
        profit_no_battery = mining_hours_no_battery * miner_profit_per_h_usd * dollar_price
        profit_battery = mining_hours_battery * miner_profit_per_h_usd * dollar_price
        profit_with_without_battery.append([profit_battery, profit_no_battery])

    return profit_with_battery, profit_without_battery, profit_with_second_miner


def calc_roi_for_miner(miner, minkW, maxkW, steps):
    battery_system_price = []
    battery_roi = []
    no_battery_system_price = []
    no_battery_roi = []
    battery_price = []
    step = (maxkW-minkW)/steps
    for i in range(steps):
        solar_kW = minkW + i*step
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
    print(miner['name'],'Max ROI:', max_roi, 'system price:', battery_system_price[max_index],
          'battery price:', battery_price[max_index], 'battery size:', battery_price[max_index] / kWh_battery_price_rub,
          'solar price:', solar_price, 'solar size:', solar_price / kW_price_rub,
          'miner price:', miner['price'],
          'inverter price:', battery_system_price[max_index] - miner['price'] - solar_price - battery_price[max_index])

    return no_battery_system_price, no_battery_roi, battery_system_price, battery_roi

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
    plt.plot([minkW + i*((maxkW-minkW)/steps) for i in range(steps)], miner['bp'], label=miner['name']+'+batt')
    plt.plot([minkW + i*((maxkW-minkW)/steps) for i in range(steps)], miner['nbp'], label=miner['name'])
plt.xlabel('solar kW')
plt.title('Стоимость системы')

plt.subplot(2, 1, 2)
for miner in miners:
    plt.plot([minkW + i*((maxkW-minkW)/steps) for i in range(steps)], miner['br'], label=miner['name']+'+batt')
    plt.plot([minkW + i*((maxkW-minkW)/steps) for i in range(steps)], miner['nbr'], label=miner['name'])
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


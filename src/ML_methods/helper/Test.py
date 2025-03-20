import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import csv
from pathlib import Path
import os

def test_predictor(file_path, reservations, tables, predictor, get_best_table, features=[]):

    booking_date_as_dt = pd.to_datetime(reservations['BookingDate']).dt.date
    unique_days = booking_date_as_dt.unique()

    if features == []:
        features = ['GuestCount', 'BookingDateDayOfWeek', 'BookingDateMonth', 'BookingStartTime', 'Duration', 'EndTime']

    for i, day in enumerate(unique_days):
        st_ing = f'Looking at day {i:>3} / {len(unique_days):>3}\t{day=}'
        reservations_for_day = reservations.loc[booking_date_as_dt == day]
        rejections = 0

        diary = np.zeros((len(tables), 64))

        for j in range(len(reservations_for_day)):

            print(f'{st_ing} \t res - {j:>3} / {len(reservations_for_day):>3}', end='\r')

            reservation = reservations_for_day.iloc[j]
            
            best_table_index = get_best_table(predictor, reservation[features], diary, tables)
            if best_table_index == -1:
                rejections += 1
                continue

            booking_code = str(reservations.loc[reservations.index == reservation.name].iloc[0]['BookingCode'])
            for d in range(int(reservation['Duration'])):
                diary[best_table_index, int(reservation['BookingStartTime']) + d] = booking_code
            
        write_schedule(file_path, diary, tables['TableCode'].tolist(), day, len(reservations_for_day), rejections)


def write_schedule(file_path, diary, tables, day, num_reservations, num_rejections):
    Path(f'{os.getcwd()}/src/outputs/{file_path}').mkdir(parents=True, exist_ok=True)
    with open(f'{os.getcwd()}/src/outputs/{file_path}/{day.strftime("%Y-%m-%d")}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        output = [['reservations: ', num_reservations, 'rejections:', num_rejections, 'wasted slots', get_wasted_slots(diary)],[]]
        for table_code, table in zip(tables, diary):
            table_output = []
            table_output.append(table_code)
            for slot in table:
                if slot == None:
                    table_output.append('')
                else:
                    table_output.append(slot)
            output.append(table_output)

        writer.writerows(output)

def get_wasted_slots(diary):
    min_booking_length = 6
    total_wasted_slots = 0
    for table in diary:
        wasted_slots = 0
        for slot in table:
            if slot == None:
                wasted_slots += 1
            else:
                total_wasted_slots += wasted_slots % min_booking_length
                wasted_slots = 0

    return total_wasted_slots
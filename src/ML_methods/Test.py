import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import csv

def test_predictor(file_path, reservations, tables, predictor, get_best_table):

    booking_date_as_dt = pd.to_datetime(reservations['BookingDate']).dt.date
    unique_days = booking_date_as_dt.unique()

    features = ['GuestCount', 'BookingDateDayOfWeek', 'BookingDateMonth', 'BookingTime', 'Duration', 'EndTime']

    for i, day in enumerate(unique_days):
        print(f'Looking at day {i} / {len(unique_days)}\t{day=}', end='\r')
        reservations_for_day = reservations.loc[booking_date_as_dt == day]
        rejections = 0

        diary = []
        for i in range(len(tables)):
            diary.append([None] * 64)

        for i in range(len(reservations_for_day)):
            reservation = reservations_for_day.iloc[i]
            
            best_table_index = get_best_table(predictor, reservation[features], diary)
            if best_table_index == -1:
                rejections += 1
                continue

            booking_code = str(reservations.loc[reservations.index == reservation.name].iloc[0]['BookingCode'])
            for i in range(int(reservation['Duration'])):
                diary[best_table_index][int(reservation['BookingTime']) + i] = booking_code
            
        write_schedule(file_path, diary, tables['TableCode'].tolist(), day, len(reservations_for_day), rejections)


def write_schedule(file_path, diary, tables, day, num_reservations, num_rejections):
    with open(f'C:/git/UoB.Y4.Dissertation/src/outputs/{file_path}/{day.strftime("%Y-%m-%d")}.csv', 'w', newline='') as csvfile:
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
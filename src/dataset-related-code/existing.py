import numpy as np
import pandas as pd
import csv
from pathlib import Path
import os

def write_schedule(file_path, diary, tables, day, num_reservations, num_rejections):

    Path(f'{os.getcwd()}/src/outputs/{file_path}').mkdir(parents=True, exist_ok=True)

    with open(f'{os.getcwd()}/src/outputs/{file_path}/{day.strftime("%Y-%m-%d")}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        output = [['reservations: ', num_reservations, 'rejections:', num_rejections, 'wasted slots', get_wasted_slots(diary)],[]]
        for table_code, table in zip(tables, diary):
            table_output = []
            table_output.append(table_code)
            for slot in table:
                if slot == 0:
                    table_output.append('')
                else:
                    table_output.append(slot)
            output.append(table_output)

        writer.writerows(output)

def get_wasted_slots(diary):
        min_booking_length = 6
        wasted_count = 0
        for table in diary:
            empty_slots = 0
            for slot in table:
                if slot != None:
                    if empty_slots < min_booking_length and empty_slots > 0:
                        wasted_count += 1
                    empty_slots = 0
                    continue
                empty_slots += 1

        return wasted_count  

# Load data
for restaurant_name in range(1,6):
    train_reservations = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-train.csv')
    test_reservations = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-test.csv')
    tables = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-tables.csv')

    test_reservations['BookingDate'] = pd.to_datetime(test_reservations['BookingDate'])

    test_days = test_reservations["BookingDate"].dt.date.unique()

    for day in test_days:
        reservations_per_day = test_reservations.loc[test_reservations['BookingDate'].dt.date == day]
        diary = [[None] * 64 for _ in range(len(tables))]

        for i, reservation in reservations_per_day.iterrows():
            table_index = tables.loc[tables['TableCode'] == reservation['TableCode']].iloc[0].name
            start_slot = (reservation['BookingStartTime'] - 36000) // (60*15)
            duration = reservation['Duration'] // (60*15)

            for i in range(duration):
                diary[table_index][start_slot+i] = reservation['BookingCode']

        write_schedule(f'Restaurant-{restaurant_name}/Existing', diary, tables['TableCode'].tolist(), day, len(reservations_per_day), 0)



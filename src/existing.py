import numpy as np
import pandas as pd
import csv

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

# Load data
restaurant_name = 1
reservations = pd.read_csv(f'C:/git/UoB.Y4.Dissertation/src/Restaurant-{restaurant_name}/reservations.csv')
existing = pd.read_csv(f'C:/git/UoB.Y4.Dissertation/src/Restaurant-{restaurant_name}/existing.csv')
tables = pd.read_csv(f'C:/git/UoB.Y4.Dissertation/src/Restaurant-{restaurant_name}/tables.csv')


# Merge reservations with existing
reservations = reservations.merge(existing, on='BookingCode', suffixes=("_left", "_right")).drop(columns=['GuestCount_right', 'BookingDate_right', 'BookingTime_right', 'Duration_right'])
reservations.columns = reservations.columns.str.replace("_left", "", regex=False)
reservations = reservations.merge(tables, on="TableCode", how="left").drop(columns=['SiteCode', 'MinCovers', 'MaxCovers'])

reservations['BookingDate'] = pd.to_datetime(reservations['BookingDate'])

unique_days = reservations["BookingDate"].dt.date.unique()

split_idx = int(len(unique_days) * 0.7)
test_days = unique_days[split_idx:]

for day in test_days:
    reservations_per_day = reservations.loc[reservations['BookingDate'].dt.date == day]
    diary = [[None] * 64 for _ in range(len(tables))]

    for i, reservation in reservations_per_day.iterrows():
        table_index = tables.loc[tables['TableCode'] == reservation['TableCode']].iloc[0].name
        start_slot = (reservation['BookingTime'] - 36000) // (60*15)
        duration = reservation['Duration'] // (60*15)

        for i in range(duration):
            diary[table_index][start_slot+i] = reservation['BookingCode']

            write_schedule(f'Restaurant-{restaurant_name}/Existing', diary, tables['TableCode'].tolist(), day, len(reservations_per_day), 0)



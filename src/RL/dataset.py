import pandas as pd
import numpy as np

def timing_fit(table, diary, start_slot, duration):
    for i in range(duration):
        if diary[table.name][start_slot + i] != None:
            return False
    return True

def tables_can_fit(tables, reservation, diary, start_slot, duration):
    size = (tables['MinCovers'] <= reservation['GuestCount']) & (reservation['GuestCount'] <= tables['MaxCovers'])
    timing = tables.loc[size].apply(lambda table: timing_fit(table, diary, start_slot, duration), axis=1)

    return size & timing

def convert_to_slot(time):
    return time // (60*15)

# Import the data
reservations = pd.read_csv('C:/git/UoB.Y4.Dissertation/src/Restaurant-1/reservations.csv')
existing =  pd.read_csv('C:/git/UoB.Y4.Dissertation/src/Restaurant-1/existing.csv')
tables = pd.read_csv('C:/git/UoB.Y4.Dissertation/src/Restaurant-1/tables.csv')

# Merge existing with reservations to get CreatedOn
reservations = pd.merge(reservations, existing, on='BookingCode', suffixes=("_left", "_right"))
reservations = reservations.drop('GuestCount_right', axis=1).drop('BookingDate_right', axis=1).drop('BookingTime_right', axis=1).drop('Duration_right', axis=1)

reservations.columns = reservations.columns.str.replace("_left", "", regex=False)

# Add the assigned table
reservations = reservations.merge(tables, on="TableCode", how="left")

# Drop what is not needed
reservations = reservations.drop('SiteCode', axis=1)
reservations = reservations.drop('MinCovers', axis=1)
reservations = reservations.drop('MaxCovers', axis=1)

# Change BookingDate from string to datetime
reservations['BookingDate'] = pd.to_datetime(reservations['BookingDate'])
reservations['CreatedOn'] = pd.to_datetime(reservations['CreatedOn'])

# Get unique days
unique_days = reservations["BookingDate"].dt.date.unique()

# Create extra rows
for day in unique_days:

    print(f'Looking at {day}')

    reservations_per_day = reservations.loc[reservations['BookingDate'].dt.date == day]
    reservations_per_day.sort_values(by='CreatedOn')

    diary = []
    for _ in range(len(tables)):
        diary.append([None] * 64)
    
    for _, reservation in reservations_per_day.iterrows():

        table_index = tables.index[tables['TableCode'] == reservation['TableCode']].tolist()[0]
        start_slot = convert_to_slot(reservation['BookingTime'] - 36000)
        duration = convert_to_slot(reservation['Duration'])

        tables_that_fit = tables_can_fit(tables, reservation, diary, start_slot, duration)

        for i, table in enumerate(tables_that_fit):
            table_code = str(tables.iloc[i]['TableCode'])

            is_free_str = f'is{table_code}Free'
            reservation[is_free_str] = table

            min_covers = f'{table_code}MinCovers'
            reservation[min_covers] = tables.iloc[i]['MinCovers']

            max_covers = f'{table_code}MaxCovers'
            reservation[max_covers] = tables.iloc[i]['MaxCovers']

        for i in range(duration):
            diary[table_index][start_slot + i] = reservation['BookingCode']



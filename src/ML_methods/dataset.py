import pandas as pd
import numpy as np

def timing_fit(table, diary, start_slot, duration):
    return all(diary[table][start_slot:start_slot + duration] == None)

def tables_can_fit(tables, reservation, diary, start_slot, duration):
    size_mask = (tables['MinCovers'] <= reservation['GuestCount']) & (reservation['GuestCount'] <= tables['MaxCovers'])
    available_tables = np.array([timing_fit(i, diary, start_slot, duration) for i in range(len(tables))])
    return size_mask & available_tables


restaurant_opening_time = 36000
restaurant_name = 'Restaurant-1'

# Load data
reservations = pd.read_csv(f'C:/git/UoB.Y4.Dissertation/src/{restaurant_name}/reservations.csv')
existing = pd.read_csv(f'C:/git/UoB.Y4.Dissertation/src/{restaurant_name}/existing.csv')
tables = pd.read_csv(f'C:/git/UoB.Y4.Dissertation/src/{restaurant_name}/tables.csv')

# Merge reservations with existing
reservations = reservations.merge(existing, on='BookingCode', suffixes=("_left", "_right")).drop(columns=['GuestCount_right', 'BookingDate_right', 'BookingTime_right', 'Duration_right'])
reservations.columns = reservations.columns.str.replace("_left", "", regex=False)
reservations = reservations.merge(tables, on="TableCode", how="left").drop(columns=['SiteCode', 'MinCovers', 'MaxCovers'])

# Convert to datetime
reservations['BookingDate'] = pd.to_datetime(reservations['BookingDate'])
reservations['CreatedOn'] = pd.to_datetime(reservations['CreatedOn'])

# Add the new columns before starting the loop
for table in tables['TableCode']:
    reservations[f'is{table}Free'] = False
    reservations[f'{table}MinCovers'] = 0
    reservations[f'{table}MaxCovers'] = 0

# Get unique days
unique_days = reservations["BookingDate"].dt.date.unique()

# Process each day
for day in unique_days:
    print(f'Looking at {day} \t --> \t {unique_days.index(day)+1} / {len(unique_days)}')

    reservations_per_day = reservations[reservations['BookingDate'].dt.date == day].sort_values(by='CreatedOn')
    reservations_per_day['start_slot'] = (reservations_per_day['BookingTime'] - restaurant_opening_time) // (60 * 15)
    reservations_per_day['duration'] = reservations_per_day['Duration'] // (60 * 15)

    diary = np.full((len(tables), 64), None, dtype=object)

    for _, reservation in reservations_per_day.iterrows():
        table_index = tables.index[tables['TableCode'] == reservation['TableCode']].tolist()[0]
        start_slot, duration = reservation['start_slot'], reservation['duration']

        tables_that_fit = tables_can_fit(tables, reservation, diary, start_slot, duration)

        for i, is_free in enumerate(tables_that_fit):
            table_code = str(tables.iloc[i]['TableCode'])
            reservations.loc[reservations['BookingCode'] == reservation['BookingCode'], f'is{table_code}Free'] = is_free
            reservations.loc[reservations['BookingCode'] == reservation['BookingCode'], f'{table_code}MinCovers'] = tables.iloc[i]['MinCovers']
            reservations.loc[reservations['BookingCode'] == reservation['BookingCode'], f'{table_code}MaxCovers'] = tables.iloc[i]['MaxCovers']

            
            reservations_per_day.loc[reservations['BookingCode'] == reservation['BookingCode'], f'is{table_code}Free'] = is_free
            reservations_per_day.loc[reservations['BookingCode'] == reservation['BookingCode'], f'{table_code}MinCovers'] = tables.iloc[i]['MinCovers']
            reservations_per_day.loc[reservations['BookingCode'] == reservation['BookingCode'], f'{table_code}MaxCovers'] = tables.iloc[i]['MaxCovers']

        diary[table_index, start_slot:start_slot + duration] = reservation['BookingCode']

    # reservations_per_day.to_csv(f'C:/git/UoB.Y4.Dissertation/src/Restaurant-1/output/output-{day}.csv')


reservations.to_csv(f'C:/git/UoB.Y4.Dissertation/src/{restaurant_name}/output.csv')

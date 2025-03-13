import os
import pandas as pd
import numpy as np
from pathlib import Path

for restaurant_name in range(1,6):

    bookings = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-bookings.csv')
    tables = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-tables.csv')

    bookings = bookings.sort_values(by='CreatedOn')

    bookings[tables['TableCode'].values] = 0 

    dates = pd.to_datetime(bookings['BookingDate']).dt.date
    unique_days = dates.unique()

    i = 0

    for date in unique_days:

        print(f'R - {restaurant_name} \t day - {i:>5} / {len(unique_days)}', end='\r')
        i+=1

        res_on_date = bookings.loc[dates == date]

        diary = np.zeros((len(tables), 64))

        for _, res in res_on_date.iterrows():
            t_idx = tables.loc[tables['TableCode'] == int(res['TableCode'])].iloc[0].name
            start_slot = (res['BookingStartTime'] - 36000) // (60*15)
            duration = res['Duration'] // (60*15)

            diary[t_idx, start_slot:start_slot+duration] = res['BookingCode']

            res_mask = (bookings['BookingCode'] == res['BookingCode']) & (bookings['TableCode'] == res['TableCode'])

            for t_i, t in enumerate(diary):
                if t_i == t_idx:
                    continue
                if tables.iloc[t_i]['MinCovers'] > res['GuestCount'] or tables.iloc[t_i]['MaxCovers'] < res['GuestCount']:
                    continue
                if (t[start_slot:start_slot+duration] != 0).any():
                    continue

                bookings.loc[res_mask,tables.iloc[t_i]['TableCode']] = 1

            prob = 1 / (bookings.loc[res_mask][tables['TableCode'].values].sum(axis=1) * 2)
            prob = prob.values[0]
            if prob == np.inf:
                prob = 0

            bookings.loc[res_mask,tables['TableCode'].values] = bookings.loc[res_mask,tables['TableCode'].values] * prob

            bookings.loc[res_mask,tables.iloc[t_idx]['TableCode']] = 0.5


    bookings = bookings.sort_values(by='BookingDate')

    
    Path(f'{os.getcwd()}/src/SQL-DATA/MLP-Soft-Encoding').mkdir(parents=True, exist_ok=True)

    bookings.to_csv(f'{os.getcwd()}/src/SQL-DATA/MLP-Soft-Encoding/Restaurant-{restaurant_name}-bookings.csv', sep=',', encoding='utf-8', index=False)
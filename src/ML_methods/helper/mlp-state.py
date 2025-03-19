import os
import pandas as pd
import numpy as np
from pathlib import Path

for restaurant_name in range(1,6):

    bookings = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-bookings.csv')
    tables = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-tables.csv')

    bookings = bookings.sort_values(by='CreatedOn')

    t_s = [f'T{t}_S{s}' for t in range(len(tables)) for s in range(64)]
    bookings = pd.concat([bookings, pd.DataFrame(0, index=bookings.index, columns=t_s)], axis=1)

    dates = pd.to_datetime(bookings['BookingDate']).dt.date
    unique_days = dates.unique()

    i = 1

    for date in unique_days:

        st_ring = f'R - {restaurant_name} \t day - {i:>5} / {len(unique_days)}'
        i+=1

        res_on_date = bookings.loc[dates == date]

        diary = np.zeros((len(tables), 64))

        j = 1

        for _, res in res_on_date.iterrows():
            print(f'{st_ring} \t reservation - {j:>3} / {len(res_on_date)}', end='\r')
            j+=1
            t_idx = tables.loc[tables['TableCode'] == int(res['TableCode'])].iloc[0].name
            start_slot = (res['BookingStartTime'] - 36000) // (60*15)
            duration = res['Duration'] // (60*15)

            res_mask = (bookings['BookingCode'] == res['BookingCode']) & (bookings['TableCode'] == res['TableCode'])

            state_at_time = (diary.flatten() != 0).astype(int)
            bookings.loc[res_mask,t_s] = state_at_time
            
            diary[t_idx, start_slot:start_slot+duration] = res['BookingCode']


    bookings = bookings.sort_values(by='BookingDate')

    
    Path(f'{os.getcwd()}/src/SQL-DATA/MLP-State').mkdir(parents=True, exist_ok=True)

    bookings.to_csv(f'{os.getcwd()}/src/SQL-DATA/MLP-State/Restaurant-{restaurant_name}-bookings.csv', sep=',', encoding='utf-8', index=False)
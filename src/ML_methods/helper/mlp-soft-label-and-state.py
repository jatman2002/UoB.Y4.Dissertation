import os
import pandas as pd
from pathlib import Path

for r in range(1,6):

    file_path = f'{os.getcwd()}/src/SQL-DATA/'
    r_name = f'Restaurant-{r}-bookings.csv'

    state_name = 'MLP-State'
    soft_label_name = 'MLP-Soft-Encoding'
    combined = 'MLP-State-Soft-Label'

    state_bookings = pd.read_csv(f'{file_path}{state_name}/{r_name}')
    soft_label_bookings = pd.read_csv(f'{file_path}{soft_label_name}/{r_name}')

    merged = state_bookings.merge(soft_label_bookings, on=['BookingCode','GuestCount','BookingDate','BookingStartTime','BookingEndTime','Duration','CreatedOn','TableCode'], how='left')

    Path(f'{file_path}{combined}').mkdir(parents=True, exist_ok=True)
    merged.to_csv(f'{file_path}{combined}/{r_name}', sep=',', encoding='utf-8', index=False)
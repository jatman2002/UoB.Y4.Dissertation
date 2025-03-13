import pandas as pd
import os

for i in range(1,6):
    bookings = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/MLP-Soft-Encoding/Restaurant-{i}-bookings.csv')

    dates = pd.to_datetime(bookings['BookingDate'])
    unique_days = dates.dt.date.unique()
    train_idx = int(len(unique_days) * 0.85)
    # val_idx = int(len(unique_days) * 0.85)

    train_days, test_days = unique_days[:train_idx], unique_days[train_idx:]


    train_data = bookings[dates.dt.date.isin(train_days)]
    # val_data = bookings[dates.dt.date.isin(val_days)]
    test_data = bookings[dates.dt.date.isin(test_days)]

    train_data.to_csv(f'{os.getcwd()}/src/SQL-DATA/MLP-Soft-Encoding/Restaurant-{i}-train.csv', sep=',', encoding='utf-8', index=False)
    # val_data.to_csv(f'C:/git/UoB.Y4.Dissertation/src/SQL-DATA/Restaurant-{i}-val.csv', sep=',', encoding='utf-8', index=False)
    test_data.to_csv(f'{os.getcwd()}/src/SQL-DATA/MLP-Soft-Encoding/Restaurant-{i}-test.csv', sep=',', encoding='utf-8', index=False)

    print(f'{i}\t{len(train_data)=}\t{len(test_data)=}')

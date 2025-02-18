import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def get_data(restaurant_name, use_label_encoder=False):

    print('LOADING DATA FROM CSV')
    reservations = pd.read_csv(f'C:/git/UoB.Y4.Dissertation/src/Restaurant-{restaurant_name}/reservations.csv')
    existing = pd.read_csv(f'C:/git/UoB.Y4.Dissertation/src/Restaurant-{restaurant_name}/existing.csv')
    tables = pd.read_csv(f'C:/git/UoB.Y4.Dissertation/src/Restaurant-{restaurant_name}/tables.csv')


    reservations = reservations.merge(existing, on='BookingCode', suffixes=("_left", "_right")).drop(columns=['GuestCount_right', 'BookingDate_right', 'BookingTime_right', 'Duration_right'])
    reservations.columns = reservations.columns.str.replace("_left", "", regex=False)
    reservations = reservations.merge(tables, on="TableCode", how="left").drop(columns=['SiteCode', 'MinCovers', 'MaxCovers'])

    print('DATA LOADED')

    # feature 
    print('MESSING AROUND WITH FEATURES')

    booking_date = pd.to_datetime(reservations['BookingDate'])

    reservations['BookingDateDayOfWeek'] = booking_date.dt.dayofweek
    reservations['BookingDateDay'] = booking_date.dt.day
    reservations['BookingDateMonth'] = booking_date.dt.month

    reservations['BookingTime'] = (reservations['BookingTime'] - 36000) / (60*15)
    reservations['Duration'] = reservations['Duration'] / (60*15)
    reservations["EndTime"] = reservations["BookingTime"] + reservations["Duration"]

    if use_label_encoder:
        label_encoder = LabelEncoder()
        reservations['TableCode'] = label_encoder.fit_transform(reservations['TableCode'])


    # Train/Test split
    print('SPLITTING DATA INTO TRAIN AND TEST')
    # Get unique days and shuffle them
    unique_days = booking_date.dt.date.unique()
    # np.random.shuffle(unique_days)

    # Split 70% for training, 30% for testing
    split_idx = int(len(unique_days) * 0.7)
    train_days, test_days = unique_days[:split_idx], unique_days[split_idx:]

    # Create train and test sets based on BookingDate
    train_data = reservations[booking_date.dt.date.isin(train_days)]
    test_data = reservations[booking_date.dt.date.isin(test_days)]

    features = ['GuestCount', 'BookingDateDayOfWeek', 'BookingDateMonth', 'BookingTime', 'Duration', 'EndTime']
    # [features.append(f(t)) for _,t in tables.iterrows() for f in (f1,f2,f3)]
    X_train, y_train = train_data[features], train_data["TableCode"]
    
    return X_train, y_train, test_data, features, restaurant_name, tables
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def feature_engineering(reservations, use_label_encoder):

    booking_date = pd.to_datetime(reservations['BookingDate'])

    reservations['BookingDateDayOfWeek'] = booking_date.dt.dayofweek
    reservations['BookingDateDay'] = booking_date.dt.day
    reservations['BookingDateMonth'] = booking_date.dt.month

    reservations['BookingStartTime'] = (reservations['BookingStartTime'] - 36000) / (60*15)
    reservations['Duration'] = reservations['Duration'] / (60*15)
    reservations["EndTime"] = reservations["BookingStartTime"] + reservations["Duration"]

    if use_label_encoder:
        label_encoder = LabelEncoder()
        reservations['TableCode'] = label_encoder.fit_transform(reservations['TableCode'])



def get_data(restaurant_name, use_label_encoder=False):

    print('LOADING DATA FROM CSV')
    train_reservations = pd.read_csv(f'C:/git/UoB.Y4.Dissertation/src/SQL-DATA/Restaurant-{restaurant_name}-train.csv')
    test_reservations = pd.read_csv(f'C:/git/UoB.Y4.Dissertation/src/SQL-DATA/Restaurant-{restaurant_name}-test.csv')
    tables = pd.read_csv(f'C:/git/UoB.Y4.Dissertation/src/SQL-DATA/Restaurant-{restaurant_name}-tables.csv')

    print('DATA LOADED')

    # feature 
    print('MESSING AROUND WITH FEATURES')

    feature_engineering(train_reservations, use_label_encoder)
    feature_engineering(test_reservations, use_label_encoder)

    features = ['GuestCount', 'BookingDateDayOfWeek', 'BookingDateMonth', 'BookingStartTime', 'Duration', 'EndTime']
    X_train, y_train = train_reservations[features], train_reservations["TableCode"]
    
    return X_train, y_train, test_reservations, features, tables
# imports
import pandas as pd
import numpy as np
import os
# import torch
# import torch.nn as nn

from keras import models
from keras.layers import Dense, Input, Dropout


from helper.Test import test_predictor
from helper.dataset import feature_engineering


def find_table(predictor, reservation, diary, tables):

    res_details = reservation.astype(float).values
    model_input = res_details.reshape(1,-1)
    probabilities = predictor(model_input, training=False)[0]
    order_of_tables = np.argsort(probabilities)[::-1]

    best_table_index = -1

    # in order of probability, find the first one that fits
    for t in order_of_tables:
        best_table = tables.iloc[t]

        # ignore where prob is 0 i.e. the classifier will never choose it
        if probabilities[t] <= 0.:
            continue

        # size constraint
        if best_table['MinCovers'] > reservation['GuestCount'] or best_table['MaxCovers'] < reservation['GuestCount']:
            continue

        # 2. time constraint
        if np.all(diary[t][int(reservation['BookingStartTime']):int(reservation['EndTime'])] != 0):
            continue

        return t

    return best_table_index

def run(restaurant_name):

    #------------------------------------------------------------------------------------------------------------------------------------

    # LOAD DATA

    tables = tables = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-tables.csv')
    train = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/MLP-Soft-Encoding/Restaurant-{restaurant_name}-train.csv')

    feature_engineering(train, False)

    features = ['GuestCount', 'BookingDateDayOfWeek', 'BookingDateMonth', 'BookingStartTime', 'Duration', 'EndTime']

    X = train.drop(columns=tables['TableCode'].astype(str).values)
    y = train[tables['TableCode'].astype(str).values]

    booking_date = pd.to_datetime(X['BookingDate']).dt.date
    unique_days = booking_date.unique()

    val_idx = int(len(unique_days) * 75 / 85)

    train_days, val_days = unique_days[:val_idx], unique_days[val_idx:]

    X_train = X[booking_date.isin(train_days)]
    X_train = X_train[features]

    val_data = train[booking_date.isin(val_days)]

    #------------------------------------------------------------------------------------------------------------------------------------

    # ONE HOT ENCODING

    # y_one_hot = pd.get_dummies(pd.DataFrame(y, columns=['TableCode']).astype(int), columns=['TableCode'])
    # y_one_hot = y_one_hot.astype(int)
    # y_one_hot = y_one_hot.reindex(columns='TableCode_'+tables['TableCode'].astype(str).values, fill_value=0)

    y_train = y[:len(X_train)]

    #------------------------------------------------------------------------------------------------------------------------------------

    # TRAIN MODEL

    print('TRAINING THE MLP CLASSIFIER')

    inp = len(features)
    output = len(tables)
    hidden1 = inp + (output - inp)//32

    inputs = Input(shape=(inp,))
    x = Dense(hidden1, activation='relu')(inputs)
    out = Dense(output, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=out)

    model.compile(optimizer='adam',
              loss='KLDivergence',
              metrics=['accuracy'])
    
    model.fit(
        X_train, 
        y_train, 
        epochs=100, 
        batch_size=10,
        verbose=1)
    #------------------------------------------------------------------------------------------------------------------------------------

    # TEST MODEL

    print('TIME TO TEST THIS THING ~~0_0~~\n')
    test_predictor(f'Restaurant-{restaurant_name}/MLP2', val_data, tables, model, find_table, features)
    print()
    print('DONE!')
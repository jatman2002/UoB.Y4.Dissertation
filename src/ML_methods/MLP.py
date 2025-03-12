# imports
import pandas as pd
import numpy as np
# import torch
# import torch.nn as nn

import tensorflow as tf
import keras
from keras import models
from keras.layers import Dense, Input

from sklearn.preprocessing import LabelEncoder

from helper.Test import test_predictor
from helper.dataset import get_data


def find_table(predictor, reservation, diary, tables):

    # probabilities = classifier.predict_proba(pd.DataFrame([reservation]))[0]
    probabilities = predictor(reservation.astype(float).values.reshape(1,-1), training=False)[0]
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
        if np.all(diary[t][int(reservation['BookingStartTime']):int(reservation['EndTime'])] != [None]*int(reservation['Duration'])):
            continue

        return t

    return best_table_index


def run(restaurant_name):

    #------------------------------------------------------------------------------------------------------------------------------------

    # LOAD DATA

    
    X, y, test_data, features, tables = get_data(restaurant_name)

    booking_date = pd.to_datetime(X['BookingDate']).dt.date
    unique_days = booking_date.unique()

    val_idx = int(len(unique_days) * 70 / 85)

    train_days, val_days = unique_days[:val_idx], unique_days[val_idx:]

    X_train = X[booking_date.isin(train_days)]
    X_val = X[booking_date.isin(val_days)]


    X_train = X_train[features]
    X_val = X_val[features]

    #------------------------------------------------------------------------------------------------------------------------------------

    # ONE HOT ENCODING

    y_one_hot = pd.get_dummies(pd.DataFrame(y, columns=['TableCode']).astype(int), columns=['TableCode'])
    y_one_hot = y_one_hot.astype(int)
    y_one_hot = y_one_hot.reindex(columns='TableCode_'+tables['TableCode'].astype(str).values, fill_value=0)

    y_train = y_one_hot[:len(X_train)]
    y_val = y_one_hot[len(X_train):]

    #------------------------------------------------------------------------------------------------------------------------------------

    # TRAIN MODEL

    print('TRAINING THE MLP CLASSIFIER')

    inp = len(features)
    hidden_1 = inp + (np.abs(len(tables) - inp)//2)
    # hidden_2 = 6 + ((np.abs(len(tables) - 6)*2)//3)
    # hidden_3 = 6 + ((np.abs(len(tables) - 6)*3)//4)
    output = len(tables)

    inputs = Input(shape=(inp,))
    x = Dense(hidden_1, activation='relu')(inputs)
    # x = Dense(hidden_2, activation='relu')(x)
    out = Dense(output, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=out)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    history = model.fit(
        X_train, 
        y_train, 
        epochs=100, 
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1)
    #------------------------------------------------------------------------------------------------------------------------------------

    # TEST MODEL

    print('TIME TO TEST THIS THING ~~0_0~~\n')
    test_predictor(f'Restaurant-{restaurant_name}/MLP', test_data, tables, model, find_table, features)
    print()
    print('DONE!')
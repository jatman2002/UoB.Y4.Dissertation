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

    
    X_train, y_train, test_data, features, tables = get_data(restaurant_name)
    X_train = X_train[features]

    #------------------------------------------------------------------------------------------------------------------------------------

    # ONE HOT ENCODING

    y_train_one_hot = pd.get_dummies(pd.DataFrame(y_train, columns=['TableCode']).astype(int), columns=['TableCode'])
    y_train_one_hot = y_train_one_hot.astype(int)
    y_train_one_hot = y_train_one_hot.reindex(columns='TableCode_'+tables['TableCode'].astype(str).values, fill_value=0)

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
    
    model.fit(X_train, y_train_one_hot, epochs=200, batch_size=32, verbose=1)
    #------------------------------------------------------------------------------------------------------------------------------------

    # TEST MODEL

    print('TIME TO TEST THIS THING ~~0_0~~\n')
    test_predictor(f'Restaurant-{restaurant_name}/MLP', test_data, tables, model, find_table, features)
    print()
    print('DONE!')
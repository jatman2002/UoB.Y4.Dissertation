# imports
import pandas as pd
import numpy as np
import os

from pathlib import Path
import pickle

from keras import models
from keras.layers import Dense, Input, BatchNormalization, Dropout

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import backend as K
import gc

from .helper.dataset import feature_engineering

import tensorflow as tf

class MLP:

    def __init__(self, restaurant_name, gpu, start_point=0):
        self.name = 'MLP'
        self.restaurant_name = restaurant_name
        self.features = ['GuestCount', 'BookingDateDayOfWeek', 'BookingDateMonth', 'BookingStartTime', 'Duration', 'EndTime']
        self.t_s = []

        self.params = {
             'hidden1': [512,1024,2046],
             'hidden2': [128,256,512],
             'dropout1': [0.1,0.2,0.3],
             'dropout2': [0.1,0.2]
        }

        self.search_space = [
            {'hidden1': h1, 'hidden2': h2, 'dropout1': d1, 'dropout2': d2} 
            for h1 in self.params['hidden1'] 
            for h2 in self.params['hidden2'] 
            for d1 in self.params['dropout1'] 
            for d2 in self.params['dropout2']
        ]

        self.start_point = start_point

        
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    def load_data(self):
        # LOAD DATA

        tables = tables = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{self.restaurant_name}-tables.csv')
        train = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/MLP-State-Soft-Label/Restaurant-{self.restaurant_name}-train.csv')

        feature_engineering(train, False)

        self.t_s = [f'T{t}_S{s}' for t in range(len(tables)) for s in range(64)]

        X = train.drop('TableCode', axis=1)
        y = train[tables['TableCode'].astype(str).values]

        booking_date = pd.to_datetime(X['BookingDate']).dt.date
        unique_days = booking_date.unique()

        val_idx = int(len(unique_days) * 75 / 85)

        train_days, val_days = unique_days[:val_idx], unique_days[val_idx:]

        X_train = X[booking_date.isin(train_days)]
        X_train = X_train[np.concatenate((self.features, self.t_s))]

        y_train = y[:len(X_train)]

        return X_train, y_train, tables
    
    def create_model(self, inp, out, params):

        inputs = Input(shape=(inp,))
        x = BatchNormalization()(inputs)
        x = Dense(params['hidden1'], activation='relu')(x)
        x = Dropout(params['dropout1'])(x)
        x = Dense(params['hidden2'], activation='relu')(x)
        x = Dropout(params['dropout2'])(x)
        x = Dense(64, activation='relu')(x)
        output = Dense(out, activation='softmax')(x)

        model = models.Model(inputs=inputs, outputs=output)

        model.compile(optimizer='adam',
                loss='KLDivergence',
                metrics=['accuracy'])
        
        return model


    def run(self):

            X_train, y_train, tables = self.load_data()

            Path(f'/mnt/fast0/jy894/models/{self.name}/grid').mkdir(parents=True, exist_ok=True)

            print('TRAINING THE MLP CLASSIFIER')

            idx = self.start_point

            es_loss = EarlyStopping(monitor='loss', patience=10, min_delta=0.001, mode='min', restore_best_weights=True)

            for params in self.search_space[self.start_point:]:

                print(f'PARAM COMBO = {idx}\t{params=}')

                model = self.create_model(len(self.features) + len(self.t_s), len(tables), params)
                
                model.fit(
                    X_train, 
                    y_train, 
                    epochs=100, 
                    batch_size=32,
                    verbose=2,
                    callbacks=[es_loss])
                
                model.save(f'/mnt/fast0/jy894/models/{self.name}/grid/{self.name}-{idx}.keras')
                
                idx += 1

                K.clear_session()
                del model
                gc.collect()
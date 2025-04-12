# imports
import pandas as pd
import numpy as np
import os

from pathlib import Path
import pickle

from keras import models
from keras.layers import Dense, Input, BatchNormalization

from .helper.dataset import feature_engineering

class MLP3:

    def __init__(self, restaurant_name, gpu):
        self.name = 'MLP3'
        self.restaurant_name = restaurant_name
        self.features = ['GuestCount', 'BookingDateDayOfWeek', 'BookingDateMonth', 'BookingStartTime', 'Duration', 'EndTime']
        self.t_s = []

    def load_data(self):
        # LOAD DATA

        tables = tables = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{self.restaurant_name}-tables.csv')
        train = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/MLP-State/Restaurant-{self.restaurant_name}-train.csv')

        feature_engineering(train, False)

        self.t_s = [f'T{t}_S{s}' for t in range(len(tables)) for s in range(64)]

        X = train.drop('TableCode', axis=1)
        y = train['TableCode']

        booking_date = pd.to_datetime(X['BookingDate']).dt.date
        unique_days = booking_date.unique()

        val_idx = int(len(unique_days) * 75 / 85)

        train_days, val_days = unique_days[:val_idx], unique_days[val_idx:]

        X_train = X[booking_date.isin(train_days)]
        X_train = X_train[np.concatenate((self.features, self.t_s))]

        #------------------------------------------------------------------------------------------------------------------------------------

        # ONE HOT ENCODING

        y_one_hot = pd.get_dummies(pd.DataFrame(y, columns=['TableCode']).astype(int), columns=['TableCode'])
        y_one_hot = y_one_hot.astype(int)
        y_one_hot = y_one_hot.reindex(columns='TableCode_'+tables['TableCode'].astype(str).values, fill_value=0)

        y_train = y_one_hot[:len(X_train)]

        return X_train, y_train, tables
    
    def create_model(self, inp, out):

        inputs = Input(shape=(inp,))
        x = BatchNormalization()(inputs)
        x = Dense(1024, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        output = Dense(out, activation='softmax')(x)

        model = models.Model(inputs=inputs, outputs=output)

        model.compile(optimizer='adam',
                loss='KLDivergence',
                metrics=['accuracy'])
        
        return model

    def run(self):

        X_train, y_train, tables = self.load_data()

        print('TRAINING THE MLP CLASSIFIER')

        model = self.create_model(len(self.features) + len(self.t_s), len(tables))
        
        history = model.fit(
            X_train, 
            y_train, 
            epochs=100, 
            batch_size=32,
                verbose=2)
        
        Path(f'{os.getcwd()}/models/{self.name}/models').mkdir(parents=True, exist_ok=True)
        Path(f'{os.getcwd()}/models/{self.name}/training').mkdir(parents=True, exist_ok=True)

        model.save(f'{os.getcwd()}/models/{self.name}/models/{self.name}-R-{self.restaurant_name}.keras')
        with open(f'{os.getcwd()}/models/{self.name}/training/{self.name}-R-{self.restaurant_name}.pkl', 'wb') as f:
            pickle.dump(history, f)
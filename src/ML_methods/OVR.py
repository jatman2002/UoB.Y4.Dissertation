# imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from Test import test_predictor
from dataset import get_data


def find_table(predictor, reservation, diary, tables):

    probabilities = predictor.predict_proba(pd.DataFrame([reservation]))[0]
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
        if np.all(diary[t][int(reservation['BookingTime']):int(reservation['EndTime'])] != [None]*int(reservation['Duration'])):
            continue

        return t

    return best_table_index


def run(restaurant_name):

    # load data from csv (raw data)

    

    X_train, y_train, test_data, features, restaurant_name, tables = get_data(restaurant_name, use_label_encoder=False)

    # fit the RF
    print('TRAINING THE OVR LOGISTIC REGRESSION CLASSIFIER')
    classifier = OneVsRestClassifier(LogisticRegression())
    classifier.fit(X_train, y_train)


    # test RF on data
    print('TIME TO TEST THIS THING ~~(0_0)~~\n')
    test_predictor(f'Restaurant-{restaurant_name}/OVR2', test_data, tables, classifier, find_table, features)
    print()
    print('DONE!')
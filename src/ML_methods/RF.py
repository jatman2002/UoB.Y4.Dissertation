# imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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
    X_train, y_train, test_data, features, tables = get_data(restaurant_name, use_label_encoder=False)

    #Hyper parameters
    n_estimators=200,
    criterion='gini',
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='sqrt',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=None,
    monotonic_cst=None

    # fit the RF
    print('TRAINING THE RANDOM FOREST CLASSIFIER')
    classifier = RandomForestClassifier(
        n_estimators=200,
        criterion='gini',
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features='sqrt',
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None
    )
    classifier.fit(X_train, y_train)


    # test RF on data
    print('TIME TO TEST THIS THING ~~0_0~~\n')
    test_predictor(f'Restaurant-{restaurant_name}/RF2', test_data, tables, classifier, find_table, features)
    print()
    print('DONE!')
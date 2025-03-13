# imports
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import LabelEncoder

from helper.Test import test_predictor
from helper.dataset import get_data



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_table(predictor, reservation, diary, tables):

    # probabilities = classifier.predict_proba(pd.DataFrame([reservation]))[0]
    probabilities = predictor(torch.tensor(reservation.astype(float).values, dtype=torch.float32).to(device)).detach().cpu().numpy()
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

    
    X_train, y_train, test_data, features, tables = get_data(restaurant_name, use_label_encoder=True)
    X_train = X_train[features]

    X_train, y_train = torch.tensor(X_train.values, dtype=torch.float32).to(device), torch.tensor(y_train.values, dtype=torch.long).to(device)

    #------------------------------------------------------------------------------------------------------------------------------------

    # TRAIN MODEL

    print('TRAINING THE MLP CLASSIFIER')

    inp = len(features)
    hidden_1 = 6 + (np.abs(len(tables) - 6)//4)
    hidden_2 = 6 + ((np.abs(len(tables) - 6)*2)//4)
    hidden_3 = 6 + ((np.abs(len(tables) - 6)*3)//4)
    output = len(tables)

    # Create MLP
    model = nn.Sequential(
        nn.Linear(inp, hidden_1),
        nn.ReLU(),
        nn.Linear(hidden_1, output),
        # nn.ReLU(),
        # nn.Linear(hidden_2, hidden_3),
        # nn.ReLU(),
        # nn.Linear(hidden_3, output),
        nn.Softmax(dim=0)
    )

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 250

    for n in range(num_epochs):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #------------------------------------------------------------------------------------------------------------------------------------

    # TEST MODEL

    print('TIME TO TEST THIS THING ~~0_0~~\n')
    test_predictor(f'Restaurant-{restaurant_name}/MLP', test_data, tables, model, find_table, features)
    print()
    print('DONE!')
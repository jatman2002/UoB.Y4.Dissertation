import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
import numpy as np
import os
import random

from RestaurantEnv import RestaurantEnv
from Test import test_predictor

class DqnNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DqnNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, input_size+(input_size-output_size)//3)
        self.l2 = nn.Linear(input_size+(input_size-output_size)//3, input_size+2*(input_size-output_size)//3)
        self.l3 = nn.Linear(input_size+2*(input_size-output_size)//3, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)
    

class ReplayMemory:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = [] # [state_t, action_t, reward_t, state_t+1, is_terminal, res_details]

    def push(self, e):
        if len(self.buffer) == self.buffer_size:
            self.buffer = self.buffer[1:]
        self.buffer.append(e)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def find_table(predictor, reservation, diary, tables):
    res_details = torch.tensor(reservation.astype(float).values, dtype=torch.float32, device=device)
    state_details = (diary.flatten() != 0).int()

    action = torch.argmax(predictor(torch.cat((res_details, state_details)))).item()

    if action == len(tables):
        return -1
    
    start = int(reservation['BookingStartTime'])
    end = int(reservation['EndTime'])

    #heavily penalise incorrect tables
    if tables.iloc[action]['MinCovers'] > reservation['GuestCount']:
        return -1
    if tables.iloc[action]['MaxCovers'] < reservation['GuestCount']:
        return -1
    if torch.any(diary[action][start:end] != 0).item():
        return -1
    
    return action


    

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)    

# Load Data
restaurant_name = 1

tables = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-tables.csv')
train = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/MLP-Soft-Encoding/Restaurant-{restaurant_name}-train.csv')

# Adjust some features
train['BookingStartTime'] = (train['BookingStartTime'] - 36000) / (60*15)
train['Duration'] = train['Duration'] / (60*15)
train["EndTime"] = train["BookingStartTime"] + train["Duration"]

features = ['GuestCount', 'BookingStartTime', 'Duration', 'EndTime']

#Create Environment
env = RestaurantEnv(tables, device)

# Define networks
input_size = len(features) + len(tables)*64
output_size = len(tables) + 1
policy_network = DqnNetwork(input_size, output_size).to(device)
target_network = DqnNetwork(input_size, output_size).to(device)
target_network.load_state_dict(policy_network.state_dict())

optimizer = optim.Adam(policy_network.parameters(), lr=1e-4, amsgrad=True)

# Extra params
epsilon = 0.5
gamma = 0.9
C = 10

# Replay Buffer
memory = ReplayMemory(1000)

booking_date = pd.to_datetime(train['BookingDate']).dt.date
unique_days = booking_date.unique()

for day in unique_days: # a day is an episode
    env.reset(device)
    total_reward = 0
    step = 0

    bookings_on_day = train.loc[booking_date == day]

    for _, reservation in bookings_on_day.iterrows():
        
        res_details = torch.tensor(reservation[features].astype(float).values, dtype=torch.float32, device=device)
        state_details = (env.state.flatten() != 0).int()

        if np.random.rand() < epsilon:
            action = torch.randint(0,len(tables)+1, (1,)).to(device)
        else:
            action = torch.argmax(policy_network(torch.cat((res_details, state_details))))

        reward = env.step(action.item(), reservation)

        if step < len(bookings_on_day)-1:
            next_res = torch.tensor(bookings_on_day.iloc[step+1][features].astype(float).values, dtype=torch.float32, device=device)
        else:
            next_res = None

        memory.push([state_details, action, reward, (env.state.flatten() != 0).int(), step == len(bookings_on_day)-1, next_res, res_details])

        step += 1

        epsilon = max(0.1, 0.99*epsilon)

        if len(memory) < 32:
            continue

        batch = memory.sample(32)

        y_j = []
        idx = 0
        for (_, _, r, s_t_1, term, n_res, _) in batch:
            if term:
                y_j.append(torch.tensor(r, dtype=torch.float32, device=device))
            else:
                y_j.append(r + gamma*torch.max(target_network(torch.cat((n_res, s_t_1)))))
            idx += 1

        x_j = [policy_network(torch.cat((res, s_t))).max() for (s_t, _, _, _, _, _, res) in batch]

        criterion = nn.SmoothL1Loss()
        loss = criterion(torch.stack(x_j).to(device), torch.stack(y_j).to(device))

        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_network.parameters(), 1.0)
        optimizer.step()

        # Update target network every C steps
        if step % C == 0:
            target_network.load_state_dict(policy_network.state_dict())

        print(f"Day {day}\t{step+1:>3}/{len(bookings_on_day):<3}", end='\r')


test_data = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-test.csv')
test_data['BookingStartTime'] = (test_data['BookingStartTime'] - 36000) / (60*15)
test_data['Duration'] = test_data['Duration'] / (60*15)
test_data["EndTime"] = test_data["BookingStartTime"] + test_data["Duration"]

print()
test_predictor(f'Restaurant-{restaurant_name}/DQN', test_data, tables, policy_network, find_table, device, features)
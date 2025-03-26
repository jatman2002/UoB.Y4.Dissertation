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
        self.l1 = nn.Linear(input_size, input_size+(input_size-output_size)//5)
        self.l2 = nn.Linear(input_size-(input_size-output_size)//5, input_size-2*(input_size-output_size)//5)
        self.l3 = nn.Linear(input_size-2*(input_size-output_size)//5, input_size-3*(input_size-output_size)//5)
        self.l4 = nn.Linear(input_size-3*(input_size-output_size)//5, input_size-4*(input_size-output_size)//5)
        self.l5 = nn.Linear(input_size-4*(input_size-output_size)//5, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)
    

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
epsilon = 0.9
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

        print(f"Day {day}\t{step+1:>3}/{len(bookings_on_day):<3}", end="\r")
        
        res_details = torch.tensor(reservation[features].astype(float).values, dtype=torch.float32, device=device)
        current_state = env.state.detach()
        state_details = (current_state.flatten() != 0).int()

        # Explore vs Exploit
        if np.random.rand() < epsilon:
            actions = torch.randperm(len(tables)+1).to(device)
        else:
            actions = torch.argsort(policy_network(torch.cat((res_details, state_details))))

        # Take the action
        action, reward = env.step(actions.tolist(), reservation)
        total_reward += reward

        if step < len(bookings_on_day)-1:
            next_res = bookings_on_day.iloc[step+1][features]
        else:
            next_res = None

        # Add to replay buffer
        memory.push([current_state, action, reward, env.state.detach(), step == len(bookings_on_day)-1, next_res, reservation[features]])

        step += 1

        # Epsilon decay
        epsilon = max(0.1, 0.9999*epsilon)

        if len(memory) < 32:
            continue

        # replay sampling
        batch = memory.sample(32)

        y_j = []
        x_j = []
        for (s_t, _, r, s_t_1, term, n_res, res) in batch:
            inp_res = torch.tensor(res.astype(float).values, dtype=torch.float32, device=device)
            inp_state = (s_t.flatten() != 0).int()
            x_j.append(policy_network(torch.cat((inp_res, inp_state)))[find_table(policy_network, res, s_t, tables)])
            if term:
                y_j.append(torch.tensor(r, dtype=torch.float32, device=device))
            else:
                policy_action = find_table(policy_network, n_res, s_t_1, tables)
                inp_n_res = torch.tensor(n_res.astype(float).values, dtype=torch.float32, device=device)
                inp_n_state = (s_t_1.flatten() != 0).int()
                y_j.append(r + gamma*(target_network(torch.cat((inp_n_res, inp_n_state)))[policy_action]))

        # Update policy network
        criterion = nn.HuberLoss()
        loss = criterion(torch.stack(x_j).to(device), torch.stack(y_j).to(device))

        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_network.parameters(), 1.0)
        optimizer.step()

        # Update target network every C steps
        if step % C == 0:
            target_network.load_state_dict(policy_network.state_dict())

    print(f"\nDay {day}\tproportional_reward={total_reward/len(bookings_on_day):>5}")


test_data = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-test.csv')
test_data['BookingStartTime'] = (test_data['BookingStartTime'] - 36000) / (60*15)
test_data['Duration'] = test_data['Duration'] / (60*15)
test_data["EndTime"] = test_data["BookingStartTime"] + test_data["Duration"]

print()
test_predictor(f'Restaurant-{restaurant_name}/DQN', test_data, tables, policy_network, find_table, device, features)
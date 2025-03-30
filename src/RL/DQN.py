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
        self.l1 = nn.Linear(input_size, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, output_size)
        # self.l4 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        # x = F.relu(self.l3(x))
        # return self.l4(x)
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

    with torch.no_grad():
        actions = torch.argsort(predictor(torch.cat((res_details, state_details))), descending=True).tolist()

        for a_i in range(len(actions)):
        
            start = int(reservation['BookingStartTime'])
            end = int(reservation['EndTime'])

            a = actions[a_i]

            #heavily penalise incorrect tables
            if tables.iloc[a]['MinCovers'] > reservation['GuestCount']:
                continue
            if tables.iloc[a]['MaxCovers'] < reservation['GuestCount']:
                continue
            if torch.any(diary[a][start:end] != 0).item():
                continue
            
            return a

        return -1

def find_y_j_table(predictor, reservation, diary, tables):
    
    res_details = torch.tensor(reservation.astype(float).values, dtype=torch.float32, device=device)
    state_details = (diary.flatten() != 0).int()

    with torch.no_grad():
        actions = torch.argsort(predictor(torch.cat((res_details, state_details))), descending=True).tolist()
        
        start = int(reservation['BookingStartTime'])
        end = int(reservation['EndTime'])

        a = actions[0]

        #heavily penalise incorrect tables
        if tables.iloc[a]['MinCovers'] > reservation['GuestCount']:
            return -1
        if tables.iloc[a]['MaxCovers'] < reservation['GuestCount']:
            return -1
        if torch.any(diary[a][start:end] != 0).item():
            return -1

        return a
    

device = torch.device(
    "cuda:1" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# print(f'{device=}\t{torch.cuda.current_device()=}')

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
output_size = len(tables)
policy_network = DqnNetwork(input_size, output_size).to(device)
target_network = DqnNetwork(input_size, output_size).to(device)
target_network.load_state_dict(policy_network.state_dict())

optimizer = optim.Adam(policy_network.parameters(), lr=1e-2, amsgrad=True)

# Extra params
epsilon = 0.9
epsilon_decay = 0.9995
gamma = 0.9
# C = 1000
batch_size = 256
TAU = 0.01

# Replay Buffer
memory = ReplayMemory(10000)

booking_date = pd.to_datetime(train['BookingDate']).dt.date
unique_days = booking_date.unique()

# total_steps = 0

actions_taken = {}
for i in range(1,len(tables)+2):
    actions_taken[i] = 0

for day in unique_days: # a day is an episode
    env.reset(device)
    total_reward = 0
    step = 0

    bookings_on_day = train.loc[booking_date == day]

    training_rejections = 0

    for _, reservation in bookings_on_day.iterrows():

        print(f"Day {day}\t{step+1:>3}/{len(bookings_on_day):<3}", end="\r")
        
        res_details = torch.tensor(reservation[features].astype(float).values, dtype=torch.float32, device=device)
        current_state = env.state.detach()
        state_details = (current_state.flatten() != 0).int()

        # Explore vs Exploit
        if np.random.rand() < epsilon:
            actions = torch.randperm(len(tables)).to(device)
        else:
            actions = torch.argsort(policy_network(torch.cat((res_details, state_details))), descending=True)

        # Take the action
        action, reward = env.step(actions.tolist(), reservation)
        total_reward += reward
        actions_taken[action+1] += 1
        if action == len(tables):
            training_rejections += 1

        if step < len(bookings_on_day)-1:
            next_res = bookings_on_day.iloc[step+1][features]
        else:
            next_res = None

        # Add to replay buffer
        memory.push([current_state, action, reward, env.state.detach(), step == len(bookings_on_day)-1, next_res, reservation[features]])

        step += 1
        # total_steps += 1

        # Epsilon decay
        epsilon = max(0.1, epsilon_decay*epsilon)

        if len(memory) < batch_size:
            continue

        # replay sampling
        batch = memory.sample(batch_size)

        s_t, a_t, r_t, s_t_1, term, n_res, res = zip(*batch)

        y_j = torch.tensor(r_t, dtype=torch.float32, device=device)
        x_j = torch.zeros(batch_size, dtype=torch.float32, device=device)

        inp_res_torch = torch.stack([torch.tensor(r.astype(float).values, dtype=torch.float32, device=device) for r in res])
        n_res_torch = torch.stack([torch.tensor(r.astype(float).values, dtype=torch.float32, device=device) if not r is None else torch.tensor([-1.0,-1.0,-1.0,-1.0], device=device) for r in n_res])

        inp_state_torch = torch.stack([(s.flatten() != 0).int() for s in s_t])
        inp_n_state_torch = torch.stack([(s.flatten() != 0).int() for s in s_t_1])

        policy_actions = torch.tensor([find_y_j_table(policy_network, r, s, tables) for s, r in zip(s_t, res)], device=device)
        rejection = policy_actions == -1
        x_j[~rejection] = policy_network(torch.cat((inp_res_torch[~rejection], inp_state_torch[~rejection]), dim=1)).gather(1, policy_actions[~rejection].unsqueeze(1)).flatten()
        x_j[rejection] = -200

        with torch.no_grad():
            term_states = torch.tensor([term for _, _, _, _, term, _, _ in batch], device=device)
            policy_n_actions = torch.tensor([find_y_j_table(policy_network, n_r, n_s, tables) if not n_r is None else -2 for n_r, n_s in zip(n_res, s_t_1)], device=device)
            rejection = policy_n_actions == -1
            invalid_actions = torch.logical_or(term_states, rejection)
            y_j[~invalid_actions] += gamma*(target_network(torch.cat((n_res_torch[~invalid_actions], inp_n_state_torch[~invalid_actions]), dim=1)).gather(1, policy_n_actions[~invalid_actions].unsqueeze(1))).flatten()
            y_j[rejection] -= 200
            
        # Update policy network
        criterion = nn.HuberLoss()
        loss = criterion(x_j, y_j)

        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        # torch.nn.utils.clip_grad_value_(policy_network.parameters(), 1.0)
        optimizer.step()

        # # Update target network every C steps
        # if total_steps % C == 0:
        #     target_network.load_state_dict(policy_network.state_dict())
        # soft update of target network
        target_net_state_dict = target_network.state_dict()
        policy_net_state_dict = policy_network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_network.load_state_dict(target_net_state_dict)

    print(f"\nDay {day}\tproportional_reward={total_reward/len(bookings_on_day):<18}\trejections = {training_rejections}")

print()
print(actions_taken)
print()

test_data = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-test.csv')
test_data['BookingStartTime'] = (test_data['BookingStartTime'] - 36000) / (60*15)
test_data['Duration'] = test_data['Duration'] / (60*15)
test_data["EndTime"] = test_data["BookingStartTime"] + test_data["Duration"]

print()
test_predictor(f'Restaurant-{restaurant_name}/DQN3', test_data, tables, policy_network, find_table, device, features)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
import os
import pickle

from RestaurantEnv import RestaurantEnv
from Test import test_predictor

# DEFINE NETWORKS
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return F.softmax(self.l3(x), dim=0)
    
class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)
    
#---------------------------------------

# DEFINE THE EPISODIC MEMORY
    
class EpisodeMemory:
    def __init__(self, episode_length, num_tables):
        self.states = torch.empty((episode_length, num_tables, 64)).to(device)
        self.actions = torch.empty((episode_length, num_tables)).to(device)
        self.rewards = torch.zeros(episode_length).to(device)
        self.n_states = torch.empty((episode_length, num_tables, 64)).to(device)
        self.values = torch.zeros(episode_length).to(device)
        self.res = torch.empty((episode_length, len(features))).to(device)
        self.next_res = [None] * episode_length

    def push(self, idx, state, action, reward, n_state, value, res, next_res):
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.n_states[idx] = n_state
        self.values[idx] = value
        self.res[idx] = res
        self.next_res[idx] = next_res

#---------------------------------------

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

#---------------------------------------

def get_advantages(rewards, values):
    advantages = torch.zeros(len(rewards)).to(device)
    last_advantage = rewards[-1]
    advantages[-1] = rewards[-1]
    for t in reversed(range(len(rewards)-1)):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        advantage = delta + gamma * lam * last_advantage
        advantages[t] = advantage
        last_advantage = advantage  

    return advantages

#---------------------------------------

device = torch.device(
    "cuda:3" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print(f'{device=}')

# Load Data
restaurant_name = 1

print('LOADING DATA')
tables = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-tables.csv')
train = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/MLP-Soft-Encoding/Restaurant-{restaurant_name}-train.csv')

# Adjust some features
train['BookingStartTime'] = (train['BookingStartTime'] - 36000) / (60*15)
train['Duration'] = train['Duration'] / (60*15)
train["EndTime"] = train["BookingStartTime"] + train["Duration"]

features = ['GuestCount', 'BookingStartTime', 'Duration', 'EndTime']

#Create Environment
print('CREATING ENV')
env = RestaurantEnv(tables, device)

# Define networks
print('CREATING NETWORKS')
input_size = len(features) + len(tables)*64
output_size = len(tables)

actor_old = PolicyNetwork(input_size, output_size).to(device)
actor = PolicyNetwork(input_size, output_size).to(device)
actor.load_state_dict(actor_old.state_dict())

critic_old = ValueNetwork(input_size).to(device)
critic = ValueNetwork(input_size).to(device)
critic.load_state_dict(critic_old.state_dict())

optimizer_actor = optim.Adam(actor.parameters(), lr=1e-3, amsgrad=True)
optimizer_critic = optim.Adam(critic.parameters(), lr=1e-3, amsgrad=True)

gamma = 0.99
epsilon = 0.2
update_iter = 5
lam = 0.95

booking_date = pd.to_datetime(train['BookingDate']).dt.date
unique_days = booking_date.unique()

trajectories = {}

print('STARTING TRAINING')
for day in unique_days: # a day is an episode
    env.reset(device)
    total_reward = 0
    step = 0

    bookings_on_day = train.loc[booking_date == day]
    memory = EpisodeMemory(len(bookings_on_day), len(tables))

    training_rejections = 0
    trajectories[day] = []

    # COLLECT TRAJECTORIES
    for _, reservation in bookings_on_day.iterrows():
        print(f"Day {day}\t{step+1:>3}/{len(bookings_on_day):<3}", end="\r")

        # Get data as tensors for network input
        res_details = torch.tensor(reservation[features].astype(float).values, dtype=torch.float32, device=device)
        current_state = env.state.detach()
        state_details = (current_state.flatten() != 0).to(device).int()

        network_input = torch.cat((res_details, state_details)).to(device)

        action_probs = actor_old(network_input).to(device)
        value = critic_old(network_input)

        sorted_action_probs = torch.argsort(action_probs, descending=True).tolist()
        action, reward = env.step(sorted_action_probs, reservation)

        total_reward += reward
        if action == len(tables):
            training_rejections += 1
        trajectories[day].append(reward)

        if step < len(bookings_on_day)-1:
            next_res = bookings_on_day.iloc[step+1][features]
        else:
            next_res = None

        memory.push(step, current_state, action_probs, reward, env.state.detach(), value, res_details, next_res)
        step += 1

    print(f"\nDay {day}\tproportional_reward={total_reward/len(bookings_on_day):<18}\trejections = {training_rejections}")

    # UPDATE AT END OF EPISODE
    returns = memory.rewards + gamma * memory.values
    inpt = torch.cat((memory.res, (memory.states.flatten(start_dim=1) != 0).int()), dim=1).to(device)
    old_policy_prob = memory.actions
    for i in range(update_iter):
        new_policy_prob = actor(inpt).to(device)
        ratio = (new_policy_prob / old_policy_prob).to(device)
        clamped = torch.clamp(ratio, 1-epsilon, 1+epsilon).to(device)
        advantages = get_advantages(memory.rewards, memory.values)
        policy_loss = (-torch.min(ratio, clamped) * advantages.reshape((len(bookings_on_day), 1))).to(device).mean()

        optimizer_actor.zero_grad()
        policy_loss.backward(retain_graph=True)
        optimizer_actor.step()
        
        value_loss_f = nn.MSELoss()
        critic_pred = critic(inpt.detach()).view_as(returns)
        value_loss = value_loss_f(critic_pred, returns.detach())

        optimizer_critic.zero_grad()
        value_loss.backward()
        optimizer_critic.step()

    actor_old.load_state_dict(actor.state_dict())
    critic_old.load_state_dict(critic.state_dict())



torch.save(actor.state_dict(), f'{os.getcwd()}/models/PPO-Actor.pt')
with open(f'{os.getcwd()}/models/DQN.pkl', 'wb') as f:
    pickle.dump(trajectories, f)

test_data = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-test.csv')
test_data['BookingStartTime'] = (test_data['BookingStartTime'] - 36000) / (60*15)
test_data['Duration'] = test_data['Duration'] / (60*15)
test_data["EndTime"] = test_data["BookingStartTime"] + test_data["Duration"]

print()
test_predictor(f'Restaurant-{restaurant_name}/DQN', test_data, tables, actor, find_table, device, features)
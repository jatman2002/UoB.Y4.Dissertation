import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
import os
import pickle

from .RestaurantEnv import RestaurantEnv

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

# DEFINE THE EPISODIC MEMORY
    
class EpisodeMemory:
    def __init__(self, episode_length, num_tables, features, device):
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


class PPO:

    def __init__(self, restaurant_name, gpu=0):
        print('---------- PPO ----------')
        self.gamma = 0.99
        self.epsilon = 0.2
        self.update_iter = 5
        self.lam = 0.95

        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        self.restaurant_name = restaurant_name

        print('LOADING DATA')
        self.train, self.tables, self.features = self.load_data()

        print('CREATING ENV')
        self.env = RestaurantEnv(self.tables, self.device)

        # Define networks
        print('CREATING NETWORKS')
        input_size = len(self.features) + len(self.tables)*64
        output_size = len(self.tables)

        self.actor_old = PolicyNetwork(input_size, output_size).to(self.device)
        self.actor = PolicyNetwork(input_size, output_size).to(self.device)
        self.actor.load_state_dict(self.actor_old.state_dict())

        self.critic_old = ValueNetwork(input_size).to(self.device)
        self.critic = ValueNetwork(input_size).to(self.device)
        self.critic.load_state_dict(self.critic_old.state_dict())

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=1e-3, amsgrad=True)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=1e-3, amsgrad=True)

    def get_advantages(self, rewards, values):
        advantages = torch.zeros(len(rewards)).to(self.device)
        last_advantage = rewards[-1]
        advantages[-1] = rewards[-1]
        for t in reversed(range(len(rewards)-1)):
            delta = rewards[t] + self.gamma * values[t+1] - values[t]
            advantage = delta + self.gamma * self.lam * last_advantage
            advantages[t] = advantage
            last_advantage = advantage  

        return advantages

    def load_data(self):
        print('LOADING DATA')
        tables = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{self.restaurant_name}-tables.csv')
        train = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/MLP-Soft-Encoding/Restaurant-{self.restaurant_name}-train.csv')

        # Adjust some features
        train['BookingStartTime'] = (train['BookingStartTime'] - 36000) / (60*15)
        train['Duration'] = train['Duration'] / (60*15)
        train["EndTime"] = train["BookingStartTime"] + train["Duration"]

        features = ['GuestCount', 'BookingStartTime', 'Duration', 'EndTime']

        return train, tables, features
    
    def update_networks(self, memory, num_bookings):
        # UPDATE AT END OF EPISODE
        returns = memory.rewards + self.gamma * memory.values
        inpt = torch.cat((memory.res, (memory.states.flatten(start_dim=1) != 0).int()), dim=1).to(self.device)
        old_policy_prob = memory.actions
        for i in range(self.update_iter):
            new_policy_prob = self.actor(inpt).to(self.device)
            ratio = (new_policy_prob / old_policy_prob).to(self.device)
            clamped = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon).to(self.device)
            advantages = self.get_advantages(memory.rewards, memory.values)
            policy_loss = (-torch.min(ratio, clamped) * advantages.reshape((num_bookings, 1))).to(self.device).mean()

            self.optimizer_actor.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.optimizer_actor.step()
            
            value_loss_f = nn.MSELoss()
            critic_pred = self.critic(inpt.detach()).view_as(returns)
            value_loss = value_loss_f(critic_pred, returns.detach())

            self.optimizer_critic.zero_grad()
            value_loss.backward()
            self.optimizer_critic.step()

        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def run(self):

        booking_date = pd.to_datetime(self.train['BookingDate']).dt.date
        unique_days = booking_date.unique()

        trajectories = {}

        print('STARTING TRAINING')
        for day in unique_days: # a day is an episode
            self.env.reset(self.device)
            total_reward = 0
            step = 0

            bookings_on_day = self.train.loc[booking_date == day]
            memory = EpisodeMemory(len(bookings_on_day), len(self.tables), self.features, self.device)

            training_rejections = 0
            trajectories[day] = []

            # COLLECT TRAJECTORIES
            for _, reservation in bookings_on_day.iterrows():
                print_string = f"Day {day}\t{step+1:>3}/{len(bookings_on_day):<3}"

                # Get data as tensors for network input
                res_details = torch.tensor(reservation[self.features].astype(float).values, dtype=torch.float32, device=self.device)
                current_state = self.env.state.detach()
                state_details = (current_state.flatten() != 0).to(self.device).int()

                network_input = torch.cat((res_details, state_details)).to(self.device)

                action_probs = self.actor_old(network_input).to(self.device)
                value = self.critic_old(network_input)

                sorted_action_probs = torch.argsort(action_probs, descending=True).tolist()
                action, reward = self.env.step(sorted_action_probs, reservation)

                total_reward += reward
                if action == len(self.tables):
                    training_rejections += 1
                trajectories[day].append(reward)

                # print(f'{print_string}\treward - {reward}')

                if step < len(bookings_on_day)-1:
                    next_res = bookings_on_day.iloc[step+1][self.features]
                else:
                    next_res = None

                memory.push(step, current_state, action_probs, reward, self.env.state.detach(), value, res_details, next_res)
                step += 1

            print(f"Day {day}\tproportional_reward={total_reward/len(bookings_on_day):<18}\trejections = {training_rejections}")

            self.update_networks(memory, len(bookings_on_day))


        torch.save(self.actor.state_dict(), f'{os.getcwd()}/models/PPO-Actor-R-{self.restaurant_name}.pt')
        with open(f'{os.getcwd()}/models/PPO-R-{self.restaurant_name}.pkl', 'wb') as f:
            pickle.dump(trajectories, f)

# test_data = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-test.csv')
# test_data['BookingStartTime'] = (test_data['BookingStartTime'] - 36000) / (60*15)
# test_data['Duration'] = test_data['Duration'] / (60*15)
# test_data["EndTime"] = test_data["BookingStartTime"] + test_data["Duration"]

# print()
# test_predictor(f'Restaurant-{restaurant_name}/DQN', test_data, tables, actor, find_table, device, features)

import gymnasium
import numpy as np
import ale_py
import torch
from torch import nn
import cv2

gymnasium.register_envs(ale_py)

env = gymnasium.make("ALE/Pong-v5", max_episode_steps=10000)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(80 * 80, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    
    def forward(self, x):
        x = self.linear(x)
        return torch.softmax(x, dim=-1)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(80 * 80, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return nn.functional.tanh(self.linear(x))

def preprocess(observation):
    observation = observation[35:-15]
    observation = observation[::2, ::2, 0]
    background = observation == 144
    observation[background] = 0
    observation[~background] = 1
    return observation

def init_game(env, reset=False):
    if reset:
        observation, info = env.reset()
    observation, *_ = env.step(env.action_space.sample())
    frame1 = preprocess(observation)
    observation, *_ = env.step(env.action_space.sample())
    frame2 = preprocess(observation)
    state = frame2 - frame1
    state = torch.tensor(state, dtype=torch.float32).view(1, -1)
    return state, frame2

num_episodes = 100000
gamma = 0.99

device = torch.device('cpu')

actions = [0, 2, 3]

actor = Actor()
critic = Critic()

ckpt = 15900
actor_weight = torch.load(f'actor_{ckpt}.pt')
actor.load_state_dict(actor_weight)
critic_weight = torch.load(f'critic_{ckpt}.pt')
critic.load_state_dict(critic_weight)

actor.to(device)
critic.to(device)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.xavier_normal_(m.bias)
    # 也可以判断是否为conv2d，使用相应的初始化方式 
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

# actor.apply(weight_init)
# critic.apply(weight_init)

actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-5)
critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-5)

for i in range(ckpt, num_episodes):
    state, frame2 = init_game(env, True)

    rewards = []
    states = []
    log_probs = []
    win = 0
    lost = 0
    while True:
        state = state.to(device)

        states.append(state)
        action_prob = actor(state)[0]
        m = torch.distributions.Categorical(action_prob)
        action_idx = m.sample()
        log_probs.append(m.log_prob(action_idx))
        action = actions[action_idx.item()]

        observation, reward, terminated, truncated, info = env.step(action)
        # cv2.putText(observation, f'{greedy} e{i + 1}', (0, 194), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.imshow('pong', observation)
        # cv2.waitKey(1)

        frame1 = frame2
        frame2 = preprocess(observation)
        state = frame2 - frame1
        state = torch.tensor(state, dtype=torch.float32).view(1, -1)

        rewards.append(reward)

        if reward == 1:
            win += 1
        elif reward == -1:
            lost += 1

        if terminated or truncated:
            break

    print(f'episode {i+1}, score {win}-{lost}')

    returns = []
    next_return = 0
    for j in range(len(rewards)):
        if rewards[-j - 1] != 0:
            next_return = 0
        cur_return = rewards[-j - 1] + gamma * next_return
        next_return = cur_return
        returns.append(cur_return)
    returns.reverse()
    
    states = torch.concat(states, dim=0)
    returns = torch.tensor(returns, dtype=torch.float32).view(-1, 1)
    values = critic(states)
    advantage = returns - values
    critic_loss = advantage.pow(2).mean()
    log_probs = torch.stack(log_probs)
    actor_loss = (-log_probs * advantage.detach()).mean()

    actor_optim.zero_grad()
    critic_optim.zero_grad()

    critic_loss.backward()
    actor_loss.backward()

    actor_optim.step()
    critic_optim.step()
    if (i + 1) % 500 == 0:
        torch.save(actor.state_dict(), f"actor_{i + 1}.pt")
        torch.save(critic.state_dict(), f"critic_{i + 1}.pt")

    

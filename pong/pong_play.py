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

device = torch.device('cpu')

actions = [0, 2, 3]

actor = Actor()
# critic = Critic()

ckpt = 37000
actor_weight = torch.load(f'actor_{ckpt}.pt')
actor.load_state_dict(actor_weight)
# critic_weight = torch.load(f'critic_{ckpt}.pt')
# critic.load_state_dict(critic_weight)

actor.to(device)
# critic.to(device)


for i in range(num_episodes):
    state, frame2 = init_game(env, True)

    win = 0
    lost = 0
    cnt = 0
    while True:
        cnt += 1
        state = state.to(device)
        with torch.no_grad():
            action_prob = actor(state)[0]
        print(action_prob)
        action_idx = np.argmax(action_prob.cpu().detach().numpy())
        action = actions[action_idx]

        observation, reward, terminated, truncated, info = env.step(action)
        # cv2.putText(observation, f'{action_prob[action_idx].item():.4f}', (0, 194), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.imwrite(f'./{i}-{cnt}.png', observation)
        cv2.imshow('pong', observation)
        cv2.waitKey(1)

        frame1 = frame2
        frame2 = preprocess(observation)
        state = frame2 - frame1
        state = torch.tensor(state, dtype=torch.float32).view(1, -1)

        if terminated or truncated:
            break

    print(f'episode {i+1}, score {win}-{lost}')

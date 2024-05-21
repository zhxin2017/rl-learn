import random

import torch.nn
import json
import agent
import board
import model
import dataset
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

model_ = model.Evaluator(n_layer=12, dmodel=256)
optimizer = optim.Adam(model_.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()


def train_model(stat_file, batch_size, device):
    ds = dataset.Ds(stat_file)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    pbar = tqdm(dl)
    for cid, color, next_turn, probs in pbar:
        cid = cid.to(device)
        color = color.to(device)
        next_turn = next_turn.to(device)
        probs = probs.to(device)
        probs_pred = model_(cid, color, next_turn)
        loss = loss_fn(probs_pred, probs)
        loss_ = loss.item()
        pbar.set_description(f'loss: {loss_: .4f}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

import os
device = torch.device('cpu')
model_.to(device)

train_cnt = 1000
train_epoch = 4
play_cnt = 100
folder = '/Users/zx/Documents/rl-exp/xiangqi/resources'
model_files = os.listdir(folder)
model_files = [f for f in model_files if f.endswith('.pt')]
if len(model_files) > 0:
    versions = [int(f.split('.')[-2]) for f in model_files]
    latest_version = max(versions)
else:
    latest_version = -1


for i in range(train_cnt):
    if i <= latest_version:
        continue
    print(f'playing {i}')
    stat_file = f'{folder}/stat.json'
    if i == 0:
        agent_ = agent.Agent(model=None, epsilon=.1, stat_file=stat_file, device=device)
    else:
        agent_ = agent.Agent(model=model_, epsilon=.1, stat_file=stat_file, device=device)

    # played_states = random.sample(list(agent_.stat.keys()), play_cnt)
    # print('play from saved states')
    # for j in tqdm(range(play_cnt)):
    #     board_ = board.Board(state_str=played_states[j])
    #     agent_.self_play(board_, 0, show_board=False)
    #
    # print('play from beginning')
    # for j in tqdm(range(play_cnt)):
    #     board_ = board.Board(first_turn=random.choice([0, 1]))
    #     agent_.self_play(board_, 0, show_board=False)

    # with open(agent_.stat_file, 'w') as f:
    #     f.write(json.dumps(agent_.stat, indent=4))
    print(f'training')
    for j in range(train_epoch):
        print(f'epoch {j}')
        train_model(stat_file, batch_size=4, device=device)
    torch.save(model_.state_dict(), f'{folder}/evaluator.{i}.pt')

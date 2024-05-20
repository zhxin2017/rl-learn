import torch.nn
import agent
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
    dl = DataLoader(ds, batch_size=batch_size)
    pbar = tqdm(dl)
    for category, color, next_turn, probs in pbar:
        category = category.to(device)
        color = color.to(device)
        next_turn = next_turn.to(device)
        probs = probs.to(device)
        probs_pred = model_(category, color, next_turn)
        loss = loss_fn(probs_pred, probs)
        loss_ = loss.item()
        pbar.set_description(f'loss: {loss_: .4f}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

import os
device = torch.device('mps')
model_.to(device)

train_cnt = 1000
play_cnt = 200
folder = '/Users/zx/Documents/rl-exp/xiangqi/resources'
model_files = os.listdir(folder)
model_files = [f for f in model_files if f.endswith('.pt')]
versions = [int(f.split('.')[-2]) for f in model_files]
latest_version = max(versions)


for i in range(train_cnt):
    if i <= latest_version:
        continue
    print(f'playing {i}')
    stat_file = f'{folder}/stat.{i}.json'
    if i == 0:
        agent_ = agent.Agent(model=None, epsilon=.7, play_cnt=play_cnt, stat_file=stat_file)
    else:
        agent_ = agent.Agent(model=model_, epsilon=.7, play_cnt=play_cnt, stat_file=stat_file)
    agent_.play()

    print(f'training')
    train_model(stat_file, batch_size=2, device=device)
    torch.save(model_.state_dict(), f'{folder}/evaluator.{i}.pt')

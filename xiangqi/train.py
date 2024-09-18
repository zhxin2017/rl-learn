import random

import torch.nn
import json
import agent
import board
import model
import dataset
from torch import optim
from torch.utils.data import DataLoader

model_ = model.Evaluator(n_layer=20, dmodel=512)
optimizer = optim.Adam(model_.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()


def train_model(stat, batch_size, device, epoch=10):
    ds = dataset.Ds(stat)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for i in range(epoch):
        cnt = 0
        for cid, color, next_turn, probs, _ in dl:
            cnt += 1
            cid = cid.to(device)
            color = color.to(device)
            next_turn = next_turn.to(device)
            probs = probs.to(device)
            probs_pred = model_(cid, color, next_turn)
            loss = loss_fn(probs_pred, probs)
            loss_ = loss.item()
            print(f'-----------batch {cnt}/{len(dl)}, loss: {loss_: .4f}------------')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

import os
train_cnt = 1000
play_cnt = 10
batch_size = 16
folder = '/Users/zx/Documents/rl-exp/xiangqi/resources'
model_files = os.listdir(folder)
model_files = [f for f in model_files if f.endswith('.pt')]
if len(model_files) > 0:
    versions = [int(f.split('.')[-2]) for f in model_files]
    latest_version = max(versions)
else:
    latest_version = -1

device = torch.device('mps')
if latest_version > -1:
    model_.load_state_dict(torch.load(f'{folder}/evaluator.{latest_version}.pt', map_location=device))

model_.to(device)


for i in range(train_cnt):
    if i + 1 <= latest_version:
        continue
    print(f'playing {i + 1}')
    model = model_
    # if i + 1 < 50:
    #     stat_file = None
    # else:
    #     stat_file = f'{folder}/stat.json'
    stat_file = f'{folder}/stat.json'
    if i == 0:
        epsilon = 1.
        play_cnt = 10000
    else:
        epsilon = .4
        play_cnt = 100
    agent_ = agent.Agent(model=model, epsilon=epsilon, stat_file=stat_file, device=device)

    saved_states = list(agent_.stat.keys())
    if len(saved_states) > play_cnt:
        played_states = random.sample(saved_states, play_cnt)
        print(f'train #{i + 1}, play from saved states')
        for j in range(play_cnt):
            board_ = board.Board(state_str=played_states[j])
            if board_.get_result() != 'draw':
                continue
            agent_.self_play(board_, 0, show_board=True)

        print(f'train #{i + 1}, train using replayed states')
        train_model(agent_.stat, batch_size=batch_size, device=device, epoch=1)

    print(f'train #{i + 1}, play from beginning')
    for j in range(play_cnt):
        print(f'==============playing game #{j + 1}/{play_cnt}==============')
        board_ = board.Board(next_turn=random.choice(['red', 'black']), me_color=random.choice(['red', 'black']))
        agent_.self_play(board_, 0, show_board=True)

    print(f'train #{i + 1}, train using new states')
    train_model(agent_.stat, batch_size=batch_size, device=device, epoch=1)

    with open(agent_.stat_file, 'w') as f:
        f.write(json.dumps(agent_.stat, indent=4))
    if (i + 1) % 1 == 0:
        torch.save(agent_.model.state_dict(), f'{folder}/evaluator.{i + 1}.pt')

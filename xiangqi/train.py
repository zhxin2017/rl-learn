import random

import torch.nn
import json
import uuid
import agent
import board
import model
import dataset
from torch import optim
from torch.utils.data import DataLoader

model_ = model.Evaluator(n_layer=20, dmodel=512)
optimizer = optim.Adam(model_.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()


def train_model(rec_file, batch_size, device, epoch=10):
    ds = dataset.Ds(rec_file)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for i in range(epoch):
        cnt = 0
        for cid, color, next_turn, probs in dl:
            cnt += 1
            cid = cid.to(device)
            color = color.to(device)
            next_turn = next_turn.to(device)
            probs = probs.to(device)
            probs_pred = model_(cid, color, next_turn)
            loss = loss_fn(probs_pred, probs)
            loss_ = loss.item()
            print(f'-----------sub_epoch #{i + 1}/{epoch}, batch #{cnt}/{len(dl)}, loss: {loss_: .4f}------------')
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

    rec_file = f'{folder}/rec.json'
    epsilon = .95**i if i < 30 else .2
    max_depth = 3 + (i + 1) // 2
    play_cnt = 2000
    sub_epoch = 2
    agent_ = agent.Agent(model=model_, epsilon=epsilon, rec_file=rec_file, device=device)

    replay_ratio = .3 if len(agent_.rec) > 0 else 0
    for j in range(play_cnt):
        print(f'==============playing game for train #{i + 1}/{train_cnt}, game #{j + 1}/{play_cnt}, epsilon {epsilon}==============')
        if random.random() < replay_ratio:
            board_ = board.Board(state_str=random.choice(agent_.rec))
        else:
            board_ = board.Board(next_turn=random.choice(['red', 'black']), me_color=random.choice(['red', 'black']))

        if board_.get_result() != 'going':
            continue
        game_uuid = uuid.uuid1().hex
        agent_.self_play(board_, show_board=False, game_uuid=game_uuid)

    agent_.save_rec()

    print(f'training #{i + 1}/{train_cnt}')
    train_model(rec_file, batch_size=batch_size, device=device, epoch=sub_epoch)

    if (i + 1) % 1 == 0:
        torch.save(agent_.model.state_dict(), f'{folder}/evaluator.{i + 1}.pt')

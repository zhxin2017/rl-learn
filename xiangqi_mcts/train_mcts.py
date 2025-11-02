import random
import torch.nn
import time
from collections import deque
import os
import pickle
import board
import model
import mcts
import numpy as np
import dataset
from torch import optim
from torch.utils.data import DataLoader, Dataset

evaluator_main = model.Evaluator(n_layer=12, dmodel=160, dhead=5)
# evaluator_oppent = model.Evaluator(n_layer=12, dmodel=160, dhead=5)
optimizer = optim.Adam(evaluator_main.parameters(), lr=1e-5)
# loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.MSELoss()


def self_play(play_num, search_num, iter_cnt):
    cid_matrices = []
    next_turns = []
    win_probs = []
    # for i in range(play_num):
    i = 0
    while i < play_num:
        i += 1
        print(f'iter {iter_cnt}, self-playing game {1}')
        my_color = random.choice(['red', 'black'])
        next_turn = random.choice(['red', 'black'])
        board_ = board.Board(next_turn=next_turn, my_color=my_color)
        result = None
        cid_matrices_per_game = []
        next_turns_per_game = []
        win_probs_per_game = []
        while True:
            node = mcts.Node(board_)
            print(f'iter {iter_cnt}, begin tree search')
            s = time.time()
            mcts.search(node, evaluator_main, search_num=search_num)
            e = time.time()
            print(f'iter {iter_cnt}, tree search finished, used {e - s:.2f} secs')
            cid_matrices_per_game.append(board_.get_cid_matrix())
            turn = 0 if board_.next_turn == 'red' else 1
            next_turns_per_game.append(turn)
            win_probs_per_game.append(float(node.W / (node.N + 1)))
            result = board_.get_result()
            if result != 'going':
                break
            a = node.select(self_play=True)
            src_row, src_col, dst_row, dst_col = node.subnodes[a].move_by
            board_.move(src_row, src_col, dst_row, dst_col)
            print(f'iter {iter_cnt}, self-playing of game {1}, step {board_.step}, choosing action {a}')
            board_.show_board(src_row, src_col, dst_row, dst_col)
        if result != 'draw':
            cid_matrices.extend(cid_matrices_per_game)
            next_turns.extend(next_turns_per_game)
            win_probs.extend(win_probs_per_game)
        else:
            i = i - 1
            print(f'iter {iter_cnt}, self-playing game {1} ended in draw, restarting')

    print('self play finished')
    return cid_matrices, next_turns, win_probs


train_num = 10000
batch_size = 32
buffer_size = 10000
num_game_per_iter = 1

data_pickle = 'data/buffer.pkl'
if os.path.exists(data_pickle):
    with open(data_pickle, 'rb') as f:
        cid_matrices_buffer, next_turns_buffer, win_probs_buffer = pickle.load(f)
else:
    cid_matrices_buffer = deque([], buffer_size)
    next_turns_buffer = deque([], buffer_size)
    win_probs_buffer = deque([], buffer_size)

ckpts = os.listdir('ckpt')
ckpts = [f for f in ckpts if f.endswith('.pt')]
if len(ckpts) == 0:
    start_num = 0
else:
    indices = [int(f[10:].split('.')[0]) for f in ckpts]
    indices.sort()
    start_num = indices[-1]
    evaluator_main.load_state_dict(torch.load(f'ckpt/evaluator_{start_num}.pt'))

for i in range(train_num):
    if i < start_num:
        continue
    # epoch = max(int(8 * 0.5**i), 1)
    epoch = 1
    search_num = min(int(100 + i), 180)
    # search_num = 2
    cid_matrices, next_turns, win_probs = self_play(num_game_per_iter, search_num, i + 1)
    cid_matrices_buffer.extend(cid_matrices)
    next_turns_buffer.extend(next_turns)
    win_probs_buffer.extend(win_probs)

    with open(data_pickle, 'wb') as f:
        pickle.dump((cid_matrices_buffer, next_turns_buffer, win_probs_buffer), f)
    
    xq_dataset = dataset.XQDataset(cid_matrices_buffer, next_turns_buffer, win_probs_buffer, aug=True)
    xq_dataloader = DataLoader(xq_dataset, batch_size=batch_size, shuffle=True)
    for e in range(epoch):
        for b, (cid_matrices_batch, next_turns_batch, win_probs_batch) in enumerate(xq_dataloader):
            pred_probs = evaluator_main(cid_matrices_batch, next_turns_batch)
            loss = loss_fn(pred_probs.view(-1), win_probs_batch.to(torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'training using replay buffer, iter {i + 1}, epoch {e + 1}, batch {b + 1}, loss {loss.item():.4f}')
    if (i + 1) % 1 == 0:
        torch.save(evaluator_main.state_dict(), f'ckpt/evaluator_{i + 1}.pt')
    

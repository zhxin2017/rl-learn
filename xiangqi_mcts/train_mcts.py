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
evaluator_oppo = model.Evaluator(n_layer=12, dmodel=160, dhead=5)
# evaluator_oppent = model.Evaluator(n_layer=12, dmodel=160, dhead=5)
optimizer = optim.Adam(evaluator_main.parameters(), lr=1e-5)
# loss_fn = torch.nn.CrossEntropyLoss()
value_loss_fn = torch.nn.MSELoss()
policy_loss_fn = torch.nn.CrossEntropyLoss(reduce='none')


def self_play(play_num, main_search_num, oppo_search_num, iter_cnt):
    cid_matrices = []
    next_turns = []
    outcomes = []
    visit_dists = []
    visit_update_mask = []
    # for i in range(play_num):
    i = 0
    while i < play_num:
        i += 1
        if len(oppo_repo) > 0:
            oppo_idx = random.choice(oppo_repo)
            evaluator_oppo.load_state_dict(torch.load(f'ckpt/evaluator_{oppo_idx}.pt'))
        else:
            oppo_idx = 0
        print(f'iter {iter_cnt}, self-playing game {i}, opponent checkpoint evaluator_{oppo_idx}.pt')
        my_color = random.choice(['red', 'black'])
        initial_next_turn = random.choice(['red', 'black'])
        main_color = random.choice(['red', 'black'])
        print(f'my color {my_color}, initial next turn {initial_next_turn}, main color {main_color}')
        board_ = board.Board(next_turn=initial_next_turn, my_color=my_color)
        result = None
        cid_matrices_per_game = []
        next_turns_per_game = []
        visit_dists_per_game = []
        visit_update_mask_per_game = []

        while True:
            result = board_.get_result()
            if result != 'going':
                if result == 'red':
                    outcome = 1
                elif result == 'black':
                    outcome = -1
                else:  # draw
                    outcome = 0
                break
            node = mcts.Node(board_)
            print(f'iter {iter_cnt}, begin tree search')
            if main_color == board_.next_turn:
                s = time.time()
                mcts.search(node, evaluator_main, search_num=main_search_num)
                e = time.time()
                print(f'iter {iter_cnt}, tree search with main evaluator finished, used {e - s:.2f} secs')
                visit_update_mask_per_game.append(1)
            else:
                s = time.time()
                mcts.search(node, evaluator_oppo, search_num=oppo_search_num)
                e = time.time()
                print(f'iter {iter_cnt}, tree search with oppo evaluator finished, used {e - s:.2f} secs')
                visit_update_mask_per_game.append(0)
            cid_matrices_per_game.append(board_.get_cid_matrix())
            next_turn = 0 if board_.next_turn == 'red' else 1
            next_turns_per_game.append(next_turn)
            a, visits_ = node.select_play()
            visits = np.zeros(90 * 90, dtype=np.float32)
            for j, v in enumerate(visits_):
                src_row, src_col, dst_row, dst_col = node.subnodes[j].move_by
                src_idx = src_row * 9 + src_col
                dst_idx = dst_row * 9 + dst_col
                act_idx = src_idx * 90 + dst_idx
                visits[act_idx] = v
            visit_dists_per_game.append(visits)
            src_row, src_col, dst_row, dst_col = node.subnodes[a].move_by
            board_.move(src_row, src_col, dst_row, dst_col)
            print(f'iter {iter_cnt}, self-playing of game {i}, step {board_.step}, choosing action {a}, with prob {visits_[a]:.4f}')
            board_.show_board()
        
        # if outcome == 0:
        #     print(f'iter {iter_cnt}, self-playing game {1} ended with a draw')
        #     i -= 1
        #     continue

        cid_matrices.extend(cid_matrices_per_game)
        next_turns.extend(next_turns_per_game)
        for _ in range(len(cid_matrices_per_game)):
            outcomes.append(outcome)
        visit_dists.extend(visit_dists_per_game)
        visit_update_mask.extend(visit_update_mask_per_game)

    print('self play finished')
    return cid_matrices, next_turns, outcomes, visit_dists, visit_update_mask


train_num = 10000
batch_size = 32
buffer_size = 10000
num_game_per_iter = 5

data_pickle = 'data/buffer.pkl'
if os.path.exists(data_pickle):
    with open(data_pickle, 'rb') as f:
        cid_matrices_buffer, next_turns_buffer, outcomes_buffer, visit_dists_buffer, visit_update_mask_buffer = pickle.load(f)
else:
    cid_matrices_buffer = deque([], buffer_size)
    next_turns_buffer = deque([], buffer_size)
    outcomes_buffer = deque([], buffer_size)
    visit_dists_buffer = deque([], buffer_size)
    visit_update_mask_buffer = deque([], buffer_size)

ckpts = os.listdir('ckpt')
ckpts = [f for f in ckpts if f.endswith('.pt')]

oppo_repo = deque([], 10)

if len(ckpts) == 0:
    start_num = 0
else:
    indices = [int(f[10:].split('.')[0]) for f in ckpts]
    indices.sort()
    start_num = indices[-1]
    oppo_repo.extend(indices[max(0, len(indices) - 10):-1])
    evaluator_main.load_state_dict(torch.load(f'ckpt/evaluator_{start_num}.pt'))


main_search_num = 200
oppo_search_num = 100

for i in range(train_num):
    if i < start_num:
        continue
    # epoch = max(int(8 * 0.5**i), 1)
    epoch = 1
    # search_num = 2
    cid_matrices, next_turns, outcomes, visit_dists, visit_update_mask = self_play(num_game_per_iter, main_search_num, oppo_search_num, i + 1)
    cid_matrices_buffer.extend(cid_matrices)
    next_turns_buffer.extend(next_turns)
    outcomes_buffer.extend(outcomes)
    visit_dists_buffer.extend(visit_dists)
    visit_update_mask_buffer.extend(visit_update_mask)

    with open(data_pickle, 'wb') as f:
        pickle.dump((cid_matrices_buffer, next_turns_buffer, outcomes_buffer, visit_dists_buffer, visit_update_mask_buffer), f)

    xq_dataset = dataset.XQDataset(cid_matrices_buffer, next_turns_buffer, outcomes_buffer, visit_dists_buffer, visit_update_mask_buffer, aug=True)
    xq_dataloader = DataLoader(xq_dataset, batch_size=batch_size, shuffle=True)
    for e in range(epoch):
        for b, (cid_matrices_batch, next_turns_batch, win_probs_batch, visit_dists_batch, visit_update_mask_batch) in enumerate(xq_dataloader):
            pred_probs, pred_act_logits = evaluator_main(cid_matrices_batch, next_turns_batch)
            value_loss = value_loss_fn(pred_probs.view(-1), win_probs_batch.to(torch.float32))
            policy_loss = (policy_loss_fn(pred_act_logits, visit_dists_batch) * visit_update_mask_batch).mean()
            loss = value_loss + policy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'training using replay buffer, iter {i + 1}, epoch {e + 1}, batch {b + 1}, policy loss {policy_loss.item():.4f}, value loss {value_loss.item():.4f}, total loss {loss.item():.4f}')
    if (i + 1) % 1 == 0:
        torch.save(evaluator_main.state_dict(), f'ckpt/evaluator_{i + 1}.pt')
    
    oppo_repo.append(i + 1)
    

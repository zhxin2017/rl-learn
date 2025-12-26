import random
import torch.nn
import time
from collections import deque
import os
import pickle
import board
import model
import mcts
import multiprocessing as mp
import numpy as np
import dataset
import actions
from torch import optim
from torch.utils.data import DataLoader, Dataset

evaluator_main = model.Evaluator(n_layer=12, dmodel=192, dhead=6)
evaluator_oppo = model.Evaluator(n_layer=12, dmodel=192, dhead=6)
ckpts = os.listdir('ckpt')
ckpts = [f for f in ckpts if f.endswith('.pt')]

if len(ckpts) == 0:
    start_num = 0
else:
    indices = [int(f[10:].split('.')[0]) for f in ckpts]
    indices.sort()
    start_num = indices[-1]
    evaluator_main.load_state_dict(torch.load(f'ckpt/evaluator_{start_num}.pt'))
    evaluator_oppo.load_state_dict(torch.load(f'ckpt/evaluator_{start_num}.pt'))

optimizer = optim.Adam(evaluator_main.parameters(), lr=1e-5)
# loss_fn = torch.nn.CrossEntropyLoss()
value_loss_fn = torch.nn.MSELoss()
policy_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

train_num = 10000
batch_size = 128
buffer_size = 10000
num_game_per_iter = 16

data_pickle = 'data/buffer.pkl'
if os.path.exists(data_pickle):
    with open(data_pickle, 'rb') as f:
        coords_buffer, next_turns_buffer, isdeads_buffer, outcomes_buffer, visits_buffer, visit_update_mask_buffer, steps_buffer = pickle.load(f)
        if len(coords_buffer) > buffer_size:
            coords_buffer = deque(list(coords_buffer)[-buffer_size:], buffer_size)
            next_turns_buffer = deque(list(next_turns_buffer)[-buffer_size:], buffer_size)
            isdeads_buffer = deque(list(isdeads_buffer)[-buffer_size:], buffer_size)
            outcomes_buffer = deque(list(outcomes_buffer)[-buffer_size:], buffer_size)
            visits_buffer = deque(list(visits_buffer)[-buffer_size:], buffer_size)
            visit_update_mask_buffer = deque(list(visit_update_mask_buffer)[-buffer_size:], buffer_size)
            steps_buffer = deque(list(steps_buffer)[-buffer_size:], buffer_size)
else:
    coords_buffer = deque([], buffer_size)
    next_turns_buffer = deque([], buffer_size)
    isdeads_buffer = deque([], buffer_size)
    outcomes_buffer = deque([], buffer_size)
    visits_buffer = deque([], buffer_size)
    visit_update_mask_buffer = deque([], buffer_size)
    steps_buffer = deque([], buffer_size)

def play_one_game(args):
    iter_cnt, game_id = args
    print(f'Starting game {game_id} in iteration {iter_cnt}...')
    main_search_num = min(int(1 + iter_cnt**0.7), 600)
    oppo_search_num = main_search_num - 50
    my_color = random.choice(['red', 'black'])
    initial_next_turn = random.choice(['red', 'black'])
    main_color = random.choice(['red', 'black'])
    board_ = board.Board(next_turn=initial_next_turn, my_color=my_color, maxstep=160)
    result = None
    coords_per_game = []
    isdeads_per_game = []
    steps_per_game = []
    next_turns_per_game = []
    visits_per_game = []
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
        if main_color == board_.next_turn:
            s = time.time()
            mcts.search(node, evaluator_main, search_num=main_search_num)
            e = time.time()
            t = e - s
            visit_update_mask_per_game.append(1)
            turn = 'main'
        else:
            s = time.time()
            mcts.search(node, evaluator_main, search_num=oppo_search_num)
            e = time.time()
            t = e - s
            visit_update_mask_per_game.append(1)
            turn = 'oppo'
        board_coords = np.array([[piece.row, piece.col] for piece in board_.pieces])
        board_isdeads = np.array([piece.isdead for piece in board_.pieces])
        coords_per_game.append(board_coords)
        isdeads_per_game.append(board_isdeads)
        steps_per_game.append(board_.step)
        next_turn = 0 if board_.next_turn == 'red' else 1
        next_turns_per_game.append(next_turn)
        a, visits_ = node.select_play()
        board_visits = np.zeros(192, dtype=np.float32)
        for j, v in enumerate(visits_):
            src_row, src_col, dst_row, dst_col = node.subnodes[j].move_by
            piece = board_.board[src_row][src_col]
            row_delta = dst_row - src_row
            col_delta = dst_col - src_col
            idx = actions.move2index[(piece.id, row_delta, col_delta)]
            board_visits[idx] = v
        visits_per_game.append(board_visits)
        src_row, src_col, dst_row, dst_col = node.subnodes[a].move_by
        board_.move(src_row, src_col, dst_row, dst_col)
        if game_id % 8 == 0:
            print(f'iter {iter_cnt}, turn: {turn}, step {board_.step}, '
                  f'mcts used {t:.4f} secs, choosing action {a}, '
                  f'with prob {visits_[a]:.4f}, max prob {np.max(visits_)}')
            board_.show_board()
    return coords_per_game, next_turns_per_game, isdeads_per_game, outcome, \
        visits_per_game, visit_update_mask_per_game, steps_per_game

if __name__ == '__main__':
    for i in range(train_num):
        if i < start_num:
            continue
        # epoch = max(int(8 * 0.5**i), 1)
        epoch = 1
        with mp.Pool(8) as pool:
            results = pool.map(play_one_game, zip([i] * num_game_per_iter, range(num_game_per_iter)))
        
        for coords, next_turns, isdeads, outcome, visits, visit_update_masks, steps in results:
            coords_buffer.extend(coords)
            next_turns_buffer.extend(next_turns)
            isdeads_buffer.extend(isdeads)
            outcomes_buffer.extend([outcome] * len(coords))
            visits_buffer.extend(visits)
            visit_update_mask_buffer.extend(visit_update_masks)
            steps_buffer.extend(steps)

        with open(data_pickle, 'wb') as f:
            pickle.dump((coords_buffer, next_turns_buffer, isdeads_buffer, outcomes_buffer, visits_buffer, visit_update_mask_buffer, steps_buffer), f)

        xq_dataset = dataset.XQDataset(coords_buffer, next_turns_buffer, isdeads_buffer, outcomes_buffer, visits_buffer, visit_update_mask_buffer, steps_buffer, aug=True)
        xq_dataloader = DataLoader(xq_dataset, batch_size=batch_size, shuffle=True)
        for e in range(epoch):
            for b, (coords_batch, next_turns_batch, outcomes_batch, visits_batch, visit_update_mask_batch, isdeads_batch, steps_batch) in enumerate(xq_dataloader):
                pred_probs, pred_act_logits = evaluator_main(coords_batch, next_turns_batch, isdeads_batch)
                value_loss = value_loss_fn(pred_probs.view(-1), outcomes_batch.to(torch.float32))
                next_turns_batch = (next_turns_batch == 0) * 1 + next_turns_batch * -1
                win_mask = (next_turns_batch == outcomes_batch) * 1.0
                policy_loss_weight = win_mask + (1 - win_mask) * 0.5
                policy_loss = (policy_loss_fn(pred_act_logits, visits_batch) * policy_loss_weight).mean()
                loss = value_loss + policy_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f'training using replay buffer, iter {i + 1}, epoch {e + 1}, batch {b + 1}/{len(xq_dataloader)}, policy loss {policy_loss.item():.4f}, value loss {value_loss.item():.4f}, total loss {loss.item():.4f}')
        if (i + 1) % 1 == 0:
            torch.save(evaluator_main.state_dict(), f'ckpt/evaluator_{i + 1}.pt')
        
        

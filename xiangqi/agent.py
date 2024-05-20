import os
import numpy as np
import torch
import sys
import random
from tqdm import tqdm
import json
from copy import deepcopy
from board import Board
sys.setrecursionlimit(10000)

class Agent:
    def __init__(self, model=None, epsilon=.7, play_cnt=10000, stat_file='stat.json', device=torch.device('mps')):
        self.epsilon = epsilon
        self.play_cnt = play_cnt
        self.stat_file = stat_file
        self.device = device
        self.model = model
        if os.path.exists(self.stat_file):
            with open(self.stat_file, 'r') as f:
                self.stat = json.loads(f.read())
        else:
            self.stat = {}

    # def start(self):
    def act(self,  board: Board):
        return self.exploit(board)

    def explore(self, board: Board):
        pieces, feasible_moves = board.feasible_moves()
        moving_index = random.choice(list(range(len(pieces))))
        feasible_moves_ = feasible_moves[moving_index]
        dst_row, dst_col = random.choice(feasible_moves_)
        src_row, src_col = pieces[moving_index].row, pieces[moving_index].col
        return src_row, src_col, dst_row, dst_col

    def exploit(self, board: Board):
        if self.model is None:
            return self.explore(board), 0
        pieces, feasible_moves = board.feasible_moves()
        n_piece = len(pieces)
        max_prob = 0
        max_pos = (0, 0, 0, 0)
        for i in range(n_piece):
            for dst_row, dst_col in feasible_moves[i]:
                board_ = deepcopy(board)
                src_row, src_col = pieces[i].row, pieces[i].col
                board_.move(src_row, src_col, dst_row, dst_col)

                category = torch.tensor(board_.board_matrix % 10, dtype=torch.int, device=self.device).view(1, 10, 9)
                color = torch.tensor(board_.board_matrix // 10, dtype=torch.int, device=self.device).view(1, 10, 9)
                next_turn = board_.next_turn
                next_turn_ = torch.tensor([next_turn], dtype=torch.int, device=self.device).view(1, 1)
                with torch.no_grad():
                    prob = torch.nn.functional.softmax(self.model(category, color, next_turn_))[0][board.next_turn]
                if prob > max_prob:
                    max_pos = (src_row, src_col, dst_row, dst_col)
                    max_prob = prob
        if max_prob == 0:
            return self.explore(board), 0
        else:
            return max_pos, max_prob

    def self_play(self, board: Board, depth):
        depth += 1
        if random.random() < self.epsilon:
            src_row, src_col, dst_row, dst_col = self.explore(board)
        else:
            (src_row, src_col, dst_row, dst_col), _ = self.exploit(board)
        board_ = deepcopy(board)
        board_.move(src_row, src_col, dst_row, dst_col)
        result = board_.get_result()
        state_str = board_.get_state()
        state_stat = self.stat.get(state_str, {'0': 0, '1': 0, '2': 0})

        if result == 3:
            if depth > 4000:
                result = 2
                state_stat[str(result)] = state_stat[str(result)] + 1
                self.stat[state_str] = state_stat
                return result
            else:
                result_ = self.self_play(board_, depth)
                state_stat[str(result_)] = state_stat[str(result_)] + 1
                self.stat[state_str] = state_stat
                return result_
        else:
            state_stat[str(result)] = state_stat[str(result)] + 1
            self.stat[state_str] = state_stat
            return result

    def play(self):
        for _ in tqdm(range(self.play_cnt)):
            board = Board(first_turn=random.choice([0, 1]))
            self.self_play(board, 0)
            # print(self.stat)
        with open(self.stat_file, 'w') as f:
            f.write(json.dumps(self.stat, indent=4))

if __name__ == '__main__':
    agent = Agent(model=None, epsilon=.7, play_cnt=1000)
    agent.play()
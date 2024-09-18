import os
import numpy as np
import dataset
from config import color_str_to_id
import torch
import sys
import random
import json
from board import Board

sys.setrecursionlimit(10000)


class Agent:
    def __init__(self, model=None, epsilon=.7, stat_file='stat.json', stat_capacity=10,
                 device=torch.device('mps')):
        self.epsilon = epsilon
        self.stat_file = stat_file
        self.device = device
        self.stat_capacity = stat_capacity
        self.model = model
        if self.stat_file is not None and os.path.exists(self.stat_file):
            with open(self.stat_file, 'r') as f:
                self.stat = json.loads(f.read())
        else:
            self.stat = {}

    def kill(self, board: Board):
        pieces, feasible_moves = board.feasible_moves()
        n_piece = len(pieces)

        kill_pos = None
        for i in range(n_piece):
            for dst_row, dst_col in feasible_moves[i]:
                src_row, src_col = pieces[i].row, pieces[i].col
                removed = board.move(src_row, src_col, dst_row, dst_col)
                if board.get_result() != 'draw':
                    kill_pos = (src_row, src_col, dst_row, dst_col)
                    board.restore(src_row, src_col, dst_row, dst_col, removed)
                    break
                else:
                    board.restore(src_row, src_col, dst_row, dst_col, removed)
            if kill_pos is not None:
                break
        return kill_pos

    def explore(self, board: Board):
        kill_pos = self.kill(board)
        if kill_pos is not None:
            src_row, src_col, dst_row, dst_col = kill_pos
            return src_row, src_col, dst_row, dst_col
        pieces, feasible_moves = board.feasible_moves()
        moving_index = random.choice(list(range(len(pieces))))
        feasible_moves_ = feasible_moves[moving_index]
        dst_row, dst_col = random.choice(feasible_moves_)
        src_row, src_col = pieces[moving_index].row, pieces[moving_index].col
        return src_row, src_col, dst_row, dst_col

    def exploit(self, board: Board):
        kill_pos = self.kill(board)
        if kill_pos is not None:
            return kill_pos, 1
        if self.model is None:
            return self.explore(board), 0
        pieces, feasible_moves = board.feasible_moves()
        n_piece = len(pieces)
        max_prob = 0
        max_pos = (0, 0, 0, 0)
        for i in range(n_piece):
            for dst_row, dst_col in feasible_moves[i]:
                src_row, src_col = pieces[i].row, pieces[i].col
                removed = board.move(src_row, src_col, dst_row, dst_col)

                category = torch.tensor(board.board_matrix % 10, dtype=torch.int, device=self.device).view(1, 10, 9)
                color_mask = category > 0
                color = torch.tensor(board.board_matrix // 10, dtype=torch.int, device=self.device).view(1, 10, 9) + 1
                color = color * color_mask + ~color_mask * 2
                board.restore(src_row, src_col, dst_row, dst_col, removed)

                next_turn_id = torch.tensor([1 - color_str_to_id[board.next_turn]], dtype=torch.int,
                                            device=self.device).view(1, 1)
                with torch.no_grad():
                    probs = torch.nn.functional.softmax(self.model(category, color, next_turn_id), dim=-1)[0]
                    prob = probs[color_str_to_id[board.next_turn]]
                if prob > max_prob:
                    max_pos = (src_row, src_col, dst_row, dst_col)
                    max_prob = prob
        if max_prob == 0:
            return self.explore(board), 0
        else:
            return max_pos, max_prob

    def self_play(self, board: Board, depth, show_board):
        depth += 1
        if random.random() < self.epsilon:
            method = 'explore'
            src_row, src_col, dst_row, dst_col = self.explore(board)
        else:
            method = 'exploit'
            (src_row, src_col, dst_row, dst_col), _ = self.exploit(board)
        removed = board.move(src_row, src_col, dst_row, dst_col)
        if show_board:
            print(f'depth {depth}, moving with {method}')
            board.show_board(src_row, src_col, dst_row, dst_col)
        result = board.get_result()
        state_str = dataset.unparse_state(board.next_turn, board.board_matrix)
        if self.stat.get(state_str) is None:
            state_stat = []
            self.stat[state_str] = state_stat
        else:
            state_stat = self.stat.get(state_str)

        if result == 'draw':
            if depth > 200:
                if len(state_stat) == self.stat_capacity:
                    state_stat.pop(0)
                state_stat.append([.5, .5])
            else:
                result = self.self_play(board, depth, show_board)
                if result != 'draw':
                    stat_ = [0, 0]
                    stat_[color_str_to_id[result]] = 1
                    if len(state_stat) == self.stat_capacity:
                        state_stat.pop(0)
                    state_stat.append(stat_)
                else:
                    if len(state_stat) == self.stat_capacity:
                        state_stat.pop(0)
                    state_stat.append([.5, .5])

        else:
            stat_ = [0, 0]
            stat_[color_str_to_id[result]] = 1
            if len(state_stat) == self.stat_capacity:
                state_stat.pop(0)
            state_stat.append(stat_)
        board.restore(src_row, src_col, dst_row, dst_col, removed)
        return result


if __name__ == '__main__':
    agent = Agent(model=None, epsilon=.7)

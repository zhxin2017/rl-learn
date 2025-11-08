import numpy as np
from typing import List
import copy
import board
import random
import torch
import dataset
from torch.utils.data import DataLoader


class Node:
    def __init__(self, board_, move_by=None):
        self.N = 0
        self.W = 0
        self.P = 0
        self.move_by = move_by
        self.supnode = None
        self.subnodes = []
        self.board_: board.Board = board_
        self.W_delta = 0

    def select_mcts(self, C_puct=5):
        values = []
        total_visit = 0

        for subnode in self.subnodes:
            total_visit += subnode.N

        total_visit_sqrt = max(total_visit**0.5, 1e-5)
        # print('showing boards of different actions')
        # cnt = 0
        for subnode in self.subnodes:
            if self.board_.next_turn == 'black':
                W = -subnode.W
            else:
                W = subnode.W
            U = C_puct * subnode.P * total_visit_sqrt / (1 + subnode.N)
            Q = W / (subnode.N + 1e-5)
            values.append(Q + U)
        a = np.argmax(values)
        return a
    
    def select_play(self):
        if self.board_.step > 30:
            tem = 0.2
        else:
            tem = 1
        
        total_visit_with_temp = 0
        total_visit = 0
        visits_with_temp = []
        visits = []
    
        kill_moves = []
        for i, subnode in enumerate(self.subnodes):
            total_visit_with_temp += subnode.N**(1 / tem)
            total_visit += subnode.N
        for i, subnode in enumerate(self.subnodes):
            visits_with_temp.append(subnode.N**(1 / tem) / total_visit_with_temp)
            visits.append(subnode.N / total_visit)
            if subnode.board_.get_result() != 'going':
                kill_moves.append(i)
        print(f'select action with prob {visits_with_temp}')
        print(f'prob with kill moves are: {[visits_with_temp[i] for i in kill_moves]}')
        a = random.choices(list(range(len(self.subnodes))), weights=visits_with_temp, k=1)[0]
        return a, visits


    def backup(self):
        node = self
        W_delta = self.W_delta
        while node is not None:
            node.W = node.W + W_delta
            node.N = node.N + 1
            node = node.supnode


def search(root: Node, evaluator, search_num=180):
    # root.W = 0
    for i in range(search_num):
        # print(f'tree search step {i + 1}')
        node = root
        while True:
            game_result = node.board_.get_result()
            # terminal state
            if game_result != 'going':
                if game_result == 'red':
                    if node.board_.next_turn == 'red':
                        node.W_delta = -1
                    else:
                        node.W_delta = 1
                elif game_result == 'black':
                    if node.board_.next_turn == 'black':
                        node.W_delta = 1
                    else:
                        node.W_delta = -1
                else:  # draw
                    node.W_delta = 0
                node.backup()
                break

            if len(node.subnodes) > 0:  # select
                a = node.select_mcts()
                node = node.subnodes[a]
            else:
                cid_matrix = torch.tensor([node.board_.get_cid_matrix()])
                next_turn = 0 if node.board_.next_turn == 'red' else 1
                next_turn = torch.tensor([next_turn])
                with torch.no_grad():
                    win_prob, act_logits = evaluator(cid_matrix, next_turn)

                act_dist = torch.softmax(act_logits, dim=-1)

                for i, (src_row, src_col) in enumerate(node.board_.feasible_srcs):
                    src_idx = src_row * 9 + src_col
                    for dst_row, dst_col in node.board_.feasible_dsts[i]:
                        dst_idx = dst_row * 9 + dst_col
                        prob = float(act_dist[0][src_idx * 90 + dst_idx])
                        new_board = copy.deepcopy(node.board_)
                        new_board.move(src_row, src_col, dst_row, dst_col)
                        # new_board.show_board()
                        new_node = Node(new_board, move_by=(src_row, src_col, dst_row, dst_col))
                        new_node.P = prob
                        new_node.supnode = node
                        node.subnodes.append(new_node)
                node.W_delta = float(win_prob[0])
                node.backup()
                break

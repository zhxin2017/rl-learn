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

    def select_mcts(self, C_puct=5):
        values = []
        visits = [subnode.N for subnode in self.subnodes]
        probs = [subnode.P for subnode in self.subnodes]
        results = [subnode.board_.get_result() for subnode in self.subnodes]
        if 'red' in results or 'black' in results:
            pass
        if self.board_.next_turn == 'black':
            Ws = [-subnode.W for subnode in self.subnodes]
        else:
            Ws = [subnode.W for subnode in self.subnodes]
        total_visit = sum(visits)

        total_visit_sqrt = total_visit**0.5
        # print('showing boards of different actions')
        # cnt = 0
        for i, subnode in enumerate(self.subnodes):
            U = C_puct * probs[i] * total_visit_sqrt / (1 + subnode.N)
            Q = Ws[i] / (subnode.N + 1e-5)
            values.append(Q + U)
        a = np.argmax(values)
        if values[a] == 0:
            a = np.argmax(probs)
        return a
    
    def select_play(self):
        if self.board_.step > 50:
            tem = 1
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
        # print(f'select action with prob {visits_with_temp}')
        print(f'prob with kill moves are: {[visits_with_temp[i] for i in kill_moves]}')
        a = random.choices(list(range(len(self.subnodes))), weights=visits_with_temp, k=1)[0]
        return a, visits


    def backup(self, W_delta):
        self.W = self.W + W_delta
        self.N = self.N + 1
        node = self.supnode
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
                    W_delta = 1
                elif game_result == 'black':
                    W_delta = -1
                else:  # draw
                    W_delta = 0
                node.backup(W_delta)
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

                act_logits_ = []

                for j, (src_row, src_col) in enumerate(node.board_.feasible_srcs):
                    src_idx = src_row * 9 + src_col
                    for dst_row, dst_col in node.board_.feasible_dsts[j]:
                        dst_idx = dst_row * 9 + dst_col
                        new_board = copy.deepcopy(node.board_)
                        new_board.move(src_row, src_col, dst_row, dst_col)
                        new_node = Node(new_board, move_by=(src_row, src_col, dst_row, dst_col))
                        new_node.supnode = node
                        result = new_board.get_result()
                        if result != 'going' and result != 'draw':
                            if result == 'red':
                                W_delta = 1
                            else:
                                W_delta = -1
                            if result == node.board_.next_turn:
                                logit = torch.tensor(1e10)
                            else:
                                logit = torch.tensor(-1e10)
                        else:
                            W_delta = 0
                            logit = act_logits[0][src_idx * 90 + dst_idx]
                        # logit = act_logits[0][src_idx * 90 + dst_idx]
                        new_node.backup(W_delta)
                        act_logits_.append(logit)
                        node.subnodes.append(new_node)
                act_logits_ = torch.stack(act_logits_)
                act_dist = torch.softmax(act_logits_, dim=-1)
                for i, new_node in enumerate(node.subnodes):
                    new_node.P = act_dist[i].item()
                W_delta = float(win_prob[0])
                node.backup(W_delta)
                break

import numpy as np
from typing import List
import copy
import board
import random
import torch

class Node:
    def __init__(self, board_, move_by=None):
        self.N = 0
        self.W = 0
        self.move_by = move_by
        self.supnode = None
        self.subnodes = []
        self.board_: board.Board = board_


    def select(self, C_puct=5, self_play=False):
        values = []
        visits = []
        total_visit = 0
        total_visit_ = 0
        W_sum = 0
        if self_play and self.board_.step > 30:
            tem = 1e04
        else:
            tem = 1

        for subnode in self.subnodes:
            total_visit += subnode.N
            total_visit_ += subnode.N**(1 / tem)
            if self.board_.next_turn == 'black':
                W = subnode.W
            else:
                W = -subnode.W
            W_sum = W_sum + np.exp(W)
        total_visit_sqrt = total_visit**0.5
        # print('showing boards of different actions')
        # cnt = 0
        for subnode in self.subnodes:
            # cnt += 1
            # print(cnt)
            # subnode.board_.show_board()
            p = np.exp(subnode.W) / W_sum
            u = C_puct * p * total_visit_sqrt / (1 + subnode.N)

            if self.board_.next_turn == 'black':
                W = subnode.W
            else:
                W = -subnode.W

            v = W / self.N 
            values.append(v + u)
            if self_play:
                visits.append(subnode.N**(1 / tem) / total_visit_)
        if self_play:
            a = random.choices(list(range(len(self.subnodes))), weights=visits, k=1)[0]
        else:
            a = np.argmax(values)
        return a
    
    def backup(self):
        self.N = 1
        W_update = self.W
        node = self.supnode
        while node is not None:
            node.W = node.W + W_update
            node.N = node.N + 1
            node = node.supnode


def search(root: Node, evaluator, search_num=180):
    for i in range(search_num):
        print(f'tree search step {i + 1}')
        node = root
        while True:
            game_result = node.board_.get_result()
            # terminal state
            if game_result != 'going':
                node.backup()
                break

            if len(node.subnodes) > 0:  # select
                a = node.select()
                node = node.subnodes[a]
            else:
                # backup
                node.N = 1
                node.backup()
                # expand and evaluate
                for i, (src_row, src_col) in enumerate(node.board_.feasible_srcs):
                    for dst_row, dst_col in node.board_.feasible_dsts[i]:
                        new_board = copy.deepcopy(node.board_)
                        new_board.move(src_row, src_col, dst_row, dst_col)
                        # new_board.show_board()
                        cids = new_board.get_cid_matrix()
                        cids = torch.tensor(cids).view(1, 10, 9)
                        next_turn = 0 if new_board.next_turn == 'red' else 1
                        next_turn = torch.tensor([[next_turn]])
                        new_node = Node(new_board, move_by=(src_row, src_col, dst_row, dst_col))
                        game_result =  new_board.get_result()
                        if game_result != 'going':
                            if game_result == 'red':
                                W = 1
                            elif game_result == 'black':
                                W = -1
                            else:
                                W = 0
                        else:
                            with torch.no_grad():
                                W = evaluator(cids, next_turn)[0].item()

                        new_node.W = W
                        new_node.supnode = node
                        node.subnodes.append(new_node)
                break

import sys

import numpy as np
import random

import torch

import agent
import board
import model


class Game:
    def __init__(self, agent=None, first_turn=0):
        self.board = board.Board(first_turn)
        self.agent = agent
        self.turn = 0

    def show_result(self, result):
        if result == 0:
            print('tie')
        if result == -1:
            print('-1 won')
        if result == 1:
            print('1 won')

    def agent_step(self):
        (src_row, src_col, dst_row, dst_col), prob = self.agent.act(self.board)
        moving = self.board.pos_to_piece[(src_row, src_col)]
        print(moving.category)
        print('win prob', prob)
        self.board.move(src_row, src_col, dst_row, dst_col)
        return src_row, src_col, dst_row, dst_col

    def person_step(self):
        move_str = input('input move str:\n')
        sub_move_strs = move_str.split('|')
        for sub_move_str in sub_move_strs:
            src_row, src_col, dst_row, dst_col = (int(s) for s in sub_move_str)
            self.board.move(src_row, src_col, dst_row, dst_col)

    def play(self):
        while True:
            print('-------------------')
            self.person_step()
            result = self.board.get_result()
            if result < 3:
                print(result)
                break
            print(self.agent_step())
            result = self.board.get_result()
            if result < 3:
                print(result)
                break


if __name__ == '__main__':
    first_turn = int(sys.argv[1])
    model_ = model.Evaluator(12, 256)
    device = torch.device('mps')
    model_.load_state_dict(torch.load('/Users/zx/Documents/rl-exp/xiangqi/evaluator.16.pt'))
    model_.to(device)
    agent_ = agent.Agent(model_, device=device)
    game = Game(agent_, first_turn=first_turn)
    game.play()
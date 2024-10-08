import os
from config import color_str_to_id
import torch
import sys
import random
from board import Board

sys.setrecursionlimit(10000)


class Agent:
    def __init__(self, model=None, epsilon=.7, rec_file='rec.json', num_last_step=3, state_capacity=500000, device=torch.device('mps')):
        self.epsilon = epsilon
        self.rec_file = rec_file
        self.device = device
        self.model = model
        self.num_last_step = num_last_step
        self.state_capacity = state_capacity
        if self.rec_file is not None and os.path.exists(self.rec_file):
            with open(self.rec_file, 'r') as f:
                self.rec = f.readlines()
            self.rec = [l.strip() for l in self.rec]
        else:
            self.rec = []

    def kill(self, board: Board):
        pieces, feasible_moves = board.feasible_moves()
        n_piece = len(pieces)

        kill_pos = None
        for i in range(n_piece):
            for dst_row, dst_col in feasible_moves[i]:
                src_row, src_col = pieces[i].row, pieces[i].col
                removed, prev_last_move = board.move(src_row, src_col, dst_row, dst_col)
                if board.get_result() != 'going':
                    kill_pos = (src_row, src_col, dst_row, dst_col)
                    board.restore(removed, prev_last_move)
                    break
                else:
                    board.restore(removed, prev_last_move)
            if kill_pos is not None:
                break
        return kill_pos

    def save_rec(self, ):
        with open(self.rec_file, 'w') as f:
            for l in self.rec[-self.state_capacity:]:
                f.write(l)
                f.write('\n')

    def explore(self, board: Board):
        kill_pos = self.kill(board)
        if kill_pos is not None:
            src_row, src_col, dst_row, dst_col = kill_pos
            return src_row, src_col, dst_row, dst_col
        pieces, feasible_moves = board.feasible_moves()
        feasible_moves_ = []
        for piece, feasible_moves_of_piece in zip(pieces, feasible_moves):
            for feasible_move_of_piece in feasible_moves_of_piece:
                feasible_moves_.append([piece, feasible_move_of_piece])
        chosen = random.choice(feasible_moves_)
        dst_row, dst_col = chosen[1]
        src_row, src_col = chosen[0].row, chosen[0].col
        return src_row, src_col, dst_row, dst_col

    def exploit(self, board: Board, try_kill=True):
        if try_kill:
            kill_pos = self.kill(board)
        else:
            kill_pos = None
        if kill_pos is not None:
            return kill_pos, 1
        if self.model is None:
            return self.explore(board), 0
        pieces, feasible_moves = board.feasible_moves()
        n_piece = len(pieces)
        max_prob = 0
        dst_pos = (0, 0, 0, 0)
        for i in range(n_piece):
            for dst_row, dst_col in feasible_moves[i]:
                src_row, src_col = pieces[i].row, pieces[i].col
                removed, prev_last_move = board.move(src_row, src_col, dst_row, dst_col)

                next_turn_id = torch.tensor([color_str_to_id[board.next_turn]], dtype=torch.int,
                                            device=self.device).view(1, 1)
                category = torch.tensor(board.cid_matrix, dtype=torch.int, device=self.device).view(1, 10, 9)
                color = torch.tensor(board.color_matrix, dtype=torch.int, device=self.device).view(1, 10, 9)
                board.restore(removed, prev_last_move)

                with torch.no_grad():
                    probs = torch.nn.functional.softmax(self.model(category, color, next_turn_id), dim=-1)[0]
                    prob = probs[color_str_to_id[board.next_turn]]

                if prob > max_prob:
                    dst_pos = (src_row, src_col, dst_row, dst_col)
                    max_prob = prob
        if max_prob == 0:
            return self.explore(board), 0
        else:
            return dst_pos, max_prob

    def self_play(self, board: Board, show_board, game_uuid, max_step=200, epsilon_decay=.99):
        epsilon_ = self.epsilon * epsilon_decay**board.step
        if random.random() < epsilon_:
            method = 'explore'
            src_row, src_col, dst_row, dst_col = self.explore(board)
        else:
            method = 'exploit'
            (src_row, src_col, dst_row, dst_col), _ = self.exploit(board)
        board.move(src_row, src_col, dst_row, dst_col)
        step = board.step
        if show_board:
            print(f'depth {step}, moving with {method}, epsilon {epsilon_}')
            board.show_board()
        result = board.get_result()
        state_str = board.dump_state()

        if result == 'going' and board.step < max_step:
            result, total_step = self.self_play(board, show_board, game_uuid, epsilon_decay=epsilon_decay)
        else:
            if result == 'going':
                result = 'draw'
            total_step = board.step

        # color | cid | next_turn | last_move
        if result == 'draw':
            result_ = '.5-.5'
        elif result == 'red':
            result_ = '1-0'
        else:
            result_ = '0-1'
        if total_step - step < self.num_last_step:
            state_str = f'{game_uuid}|{step}|{total_step}|{result_}|{state_str}'
            self.rec.append(state_str)
        return result, total_step


if __name__ == '__main__':
    agent = Agent(model=None, epsilon=.7)

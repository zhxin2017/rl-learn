from torch.utils.data import Dataset
from config import color_str_to_id
import json
import torch
import numpy as np


def parse_state(state_str):
    board, next_turn = state_str.split('|')
    board_matrix = []
    for i in range(90):
        board_matrix.append(int(board[i * 2: (i + 1) * 2]))
    board_matrix = np.array(board_matrix, dtype=int).reshape(10, 9)
    cid = board_matrix % 10
    no_color_mask = cid == 0
    color = board_matrix // 10
    color = color * (1 - no_color_mask) + no_color_mask * 2
    return cid, color, next_turn, board_matrix


class Ds(Dataset):
    def __init__(self, stat):
        self.stat = stat
        # self.stat_file = stat_file
        # with open(stat_file, 'r') as f:
        #     self.stat = json.loads(f.read())
        self.state_strs = list(self.stat.keys())

    def __getitem__(self, item):
        state_str = self.state_strs[item]
        stat_ = torch.tensor(self.stat[state_str]).sum(dim=0)
        probs = stat_ / stat_.sum()
        cid, color, next_turn, _ = parse_state(state_str)
        next_turn_id = color_str_to_id[next_turn]
        return cid, color, next_turn_id, probs

    def __len__(self):
        return len(self.state_strs)

if __name__ == '__main__':
    stat_file = '/Users/zx/Documents/rl-exp/xiangqi/stat.json'
    ds = Ds(stat_file)
    for category, color, next_turn, probs in ds:
        print(category.shape)
        print(color.shape)
        print(probs.shape)
        print(next_turn)

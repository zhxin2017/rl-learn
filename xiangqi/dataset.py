from torch.utils.data import Dataset
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
    color = board_matrix // 10
    return cid, color, int(next_turn), board_matrix


class Ds(Dataset):
    def __init__(self, stat_file):
        self.stat_file = stat_file
        with open(stat_file, 'r') as f:
            self.stat = json.loads(f.read())
        self.state_strs = list(self.stat.keys())

    def __getitem__(self, item):
        state_str = self.state_strs[item]
        stat_ = torch.tensor(self.stat[state_str]).sum(dim=0)
        probs = stat_ / stat_.sum()
        cid, color, next_turn, _ = parse_state(state_str)
        return cid, color, next_turn, probs

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

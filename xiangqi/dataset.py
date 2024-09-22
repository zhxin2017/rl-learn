from torch.utils.data import Dataset
from config import color_str_to_id
import state
import torch


def unparse_state(next_turn, board_matrix):
    board_matrix = board_matrix.reshape(-1)
    state = ''.join(['0' * (1 - b // 10) + str(b) for b in board_matrix]) + '|' + next_turn
    return state


# def flip_board(state_str):
#     cid, color, next_turn, board_matrix = parse_state(state_str)


class Ds(Dataset):
    def __init__(self, rec_file, capacity=10):
        self.rec_file = rec_file
        self.capacity = capacity
        with open(self.rec_file, 'r') as f:
            rec_lines = f.readlines()
        self.stat = {}
        self.state_keys = []
        for l in rec_lines:
            rec = l.strip()
            state_key, step, total_step, result = state.parse_state_str_for_agent(rec)
            occur = self.stat.get(state_key)
            if occur is None:
                occur = [result]
                self.stat[state_key] = occur
                self.state_keys.append(state_key)
            else:
                occur.append(result)

    def __getitem__(self, item):
        state_str = self.state_keys[item]
        stat_ = torch.tensor(self.stat[state_str])[-self.capacity:]
        probs = stat_.sum(dim=0) / stat_.sum()
        color_matrix, cid_matrix, next_turn = state.parse_state_str_for_model(state_str)
        return cid_matrix, color_matrix, next_turn, probs

    def __len__(self):
        return len(self.state_keys)


if __name__ == '__main__':
    stat_file = '/Users/zx/Documents/rl-exp/xiangqi/stat.json'
    ds = Ds(stat_file)
    for category, color, next_turn, probs in ds:
        print(category.shape)
        print(color.shape)
        print(probs.shape)
        print(next_turn)

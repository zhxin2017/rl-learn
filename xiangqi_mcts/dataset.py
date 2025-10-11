from torch.utils.data import Dataset
from config import color_str_to_id
import state
import numpy as np
import torch


# def flip_board(state_str):
#     cid, color, next_turn, board_matrix = parse_state(state_str)



class XQDataset(Dataset):
    def __init__(self, cid_matrices, next_turns, win_probs, aug=False):
        super().__init__()
        self.cid_matrices = cid_matrices
        self.next_turns = next_turns
        self.win_probs = win_probs
        if aug:
            cid_matrices_flip = []
            next_turns_flip = []
            win_probs_flip = []
            for i, m in enumerate(cid_matrices):
                cid_matrices_flip.append(np.flip(m, axis=1).copy())
                next_turns_flip.append(next_turns[i])
                win_probs_flip.append(win_probs[i])
                cid_matrices_flip.append(np.flip(m, axis=0).copy())
                next_turns_flip.append(next_turns[i])
                win_probs_flip.append(win_probs[i])
            cid_matrices_switch = []
            next_turns_switch = []
            win_probs_switch = []
            for i, m in enumerate(cid_matrices):
                cid_black_mask = m > 7
                cid_red_mask = (m > 0) * (1 - cid_black_mask)
                cid_matrix_red = cid_black_mask * m - 7
                cid_matrix_black = cid_red_mask * m + 7
                cid_matrices_switch.append(cid_matrix_red + cid_matrix_black)
                next_turns_switch.append(1 - next_turns[i])
                win_probs_switch.append(-win_probs[i])
            self.cid_matrices.extend(cid_matrices_flip)
            self.cid_matrices.extend(cid_matrices_switch)
            self.next_turns.extend(next_turns_flip)
            self.next_turns.extend(next_turns_switch)
            self.win_probs.extend(win_probs_flip)
            self.win_probs.extend(win_probs_switch)

    def __getitem__(self, index):
        return self.cid_matrices[index], self.next_turns[index], self.win_probs[index]
    
    def __len__(self):
        return len(self.cid_matrices)


if __name__ == '__main__':
    stat_file = '/Users/zx/Documents/rl-exp/xiangqi/resources/rec.txt'
    ds = Ds(stat_file)
    for category, color, next_turn, probs in ds:
        print(probs)

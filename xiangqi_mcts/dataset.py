from torch.utils.data import Dataset
from config import color_str_to_id
import state
import numpy as np
import torch


# def flip_board(state_str):
#     cid, color, next_turn, board_matrix = parse_state(state_str)

def flip(cid_matrix):
    # Flip the board horizontally
    cid_matrix_flip_hor = np.flip(cid_matrix, axis=1).copy()
    cid_matrix_flip_ver = np.flip(cid_matrix, axis=0).copy()
    cid_matrix_flip_hor_ver = np.flip(cid_matrix_flip_hor, axis=0).copy()
    return [cid_matrix_flip_hor, cid_matrix_flip_ver, cid_matrix_flip_hor_ver]

def switch(cid_matrix, next_turn, win_prob):
    cid_matrix_black_mask = cid_matrix > 7
    cid_matrix_red_mask = (cid_matrix > 0) * (1 - cid_matrix_black_mask)
    cid_matrix_red = cid_matrix_black_mask * cid_matrix - 7
    cid_matrix_black = cid_matrix_red_mask * cid_matrix + 7
    return cid_matrix_red + cid_matrix_black, 1 - next_turn, -win_prob


class XQDataset(Dataset):
    def __init__(self, cid_matrices, next_turns, win_probs, aug=False):
        super().__init__()
        self.cid_matrices = cid_matrices
        self.next_turns = next_turns
        self.win_probs = win_probs
        if aug:
            cid_matrices_aug = []
            next_turns_aug = []
            win_probs_aug = []
            for i, m in enumerate(cid_matrices):
                cid_matrices_aug.extend(flip(m))
                next_turns_aug.extend([next_turns[i]] * 3)
                win_probs_aug.extend([win_probs[i]] * 3)
                cid_matrix_switch, next_turn_switch, win_prob_switch = switch(m, next_turns[i], win_probs[i])
                cid_matrices_aug.append(cid_matrix_switch)
                next_turns_aug.append(next_turn_switch)
                win_probs_aug.append(win_prob_switch)
                cid_matrix_switch_flips = flip(cid_matrix_switch)
                cid_matrices_aug.extend(cid_matrix_switch_flips)
                next_turns_aug.extend([next_turn_switch] * 3)
                win_probs_aug.extend([win_prob_switch] * 3)

            self.cid_matrices.extend(cid_matrices_aug)
            self.next_turns.extend(next_turns_aug)
            self.win_probs.extend(win_probs_aug)

    def __getitem__(self, index):
        return self.cid_matrices[index], self.next_turns[index], self.win_probs[index]
    
    def __len__(self):
        return len(self.cid_matrices)

from torch.utils.data import Dataset
from config import color_str_to_id
import state
import numpy as np
import torch


# def flip_board(state_str):
#     cid, color, next_turn, board_matrix = parse_state(state_str)

def flip(cid_matrix, visit_dist):
    # Flip the board horizontally
    # visit_dist_matrix = visit_dist.reshape(90, 90)
    cid_matrix_flip_hor = np.flip(cid_matrix, axis=1).copy()
    cid_matrix_flip_ver = np.flip(cid_matrix, axis=0).copy()
    cid_matrix_flip_hor_ver = np.flip(cid_matrix_flip_hor, axis=0).copy()
    board_indices = np.arange(90).reshape(10, 9)
    board_indices_flip_hor = np.flip(board_indices, axis=1).copy().reshape(90, 1)
    board_indices_flip_ver = np.flip(board_indices, axis=0).copy().reshape(90, 1)
    board_indices_flip_hor_ver = np.flip(board_indices_flip_hor, axis=0).copy().reshape(90, 1)

    visit_indices_flip_hor = (board_indices_flip_hor @ board_indices_flip_hor.T).flatten()
    visit_indices_flip_ver = (board_indices_flip_ver @ board_indices_flip_ver.T).flatten()
    visit_indices_flip_hor_ver = (board_indices_flip_hor_ver @ board_indices_flip_hor_ver.T).flatten()

    visit_dist_flip_hor = visit_dist[visit_indices_flip_hor]
    visit_dist_flip_ver = visit_dist[visit_indices_flip_ver]
    visit_dist_flip_hor_ver = visit_dist[visit_indices_flip_hor_ver]
    cid_matrix_aug = [cid_matrix_flip_hor, cid_matrix_flip_ver, cid_matrix_flip_hor_ver]
    visit_dist_aug = [visit_dist_flip_hor, visit_dist_flip_ver, visit_dist_flip_hor_ver]
    return cid_matrix_aug, visit_dist_aug

def switch(cid_matrix, next_turn, outcome):
    cid_matrix_black_mask = cid_matrix > 7
    cid_matrix_red_mask = (cid_matrix > 0) * (1 - cid_matrix_black_mask)
    cid_matrix_red = cid_matrix_black_mask * cid_matrix - 7
    cid_matrix_black = cid_matrix_red_mask * cid_matrix + 7
    return cid_matrix_red + cid_matrix_black, 1 - next_turn, -outcome


class XQDataset(Dataset):
    def __init__(self, cid_matrices, next_turns, outcomes, visit_dists, aug=False):
        super().__init__()
        self.cid_matrices = cid_matrices
        self.next_turns = next_turns
        self.outcomes = outcomes
        self.visit_dists = visit_dists
        if aug:
            cid_matrices_aug = []
            next_turns_aug = []
            outcomes_aug = []
            visit_dists_aug = []
            for i, m in enumerate(cid_matrices):
                cid_matrices_flip_aug, visits_flip_aug = flip(m, visit_dists[i])
                cid_matrices_aug.extend(cid_matrices_flip_aug)
                visit_dists_aug.extend(visits_flip_aug)
                next_turns_aug.extend([next_turns[i]] * 3)
                outcomes_aug.extend([outcomes[i]] * 3)
                cid_matrix_switch, next_turn_switch, outcome_switch = switch(m, next_turns[i], outcomes[i])
                cid_matrices_aug.append(cid_matrix_switch)
                next_turns_aug.append(next_turn_switch)
                outcomes_aug.append(outcome_switch)
                visit_dists_aug.append(visit_dists[i])
                cid_matrix_switch_flips, visit_dist_switch_flips = flip(cid_matrix_switch, visit_dists[i])
                cid_matrices_aug.extend(cid_matrix_switch_flips)
                next_turns_aug.extend([next_turn_switch] * 3)
                outcomes_aug.extend([outcome_switch] * 3)
                visit_dists_aug.extend(visit_dist_switch_flips)

            self.cid_matrices.extend(cid_matrices_aug)
            self.next_turns.extend(next_turns_aug)
            self.outcomes.extend(outcomes_aug)
            self.visit_dists.extend(visit_dists_aug)

    def __getitem__(self, index):
        return self.cid_matrices[index], self.next_turns[index], self.outcomes[index], self.visit_dists[index]

    def __len__(self):
        return len(self.cid_matrices)

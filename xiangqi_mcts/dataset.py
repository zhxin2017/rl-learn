from torch.utils.data import Dataset
from config import color_str_to_id
import state
import copy
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
    board_indices_flip_hor = np.flip(board_indices, axis=1).copy()
    board_indices_flip_ver = np.flip(board_indices, axis=0).copy().reshape(90, 1)
    board_indices_flip_hor_ver = np.flip(board_indices_flip_hor, axis=0).copy().reshape(90, 1)
    board_indices_flip_hor = board_indices_flip_hor.reshape(90, 1)

    visit_indices_flip_hor = (90 * board_indices_flip_hor + board_indices_flip_hor.T).flatten()
    visit_indices_flip_ver = (90 * board_indices_flip_ver + board_indices_flip_ver.T).flatten()
    visit_indices_flip_hor_ver = (90 * board_indices_flip_hor_ver + board_indices_flip_hor_ver.T).flatten()

    visit_dist_flip_hor = visit_dist[visit_indices_flip_hor]
    visit_dist_flip_ver = visit_dist[visit_indices_flip_ver]
    visit_dist_flip_hor_ver = visit_dist[visit_indices_flip_hor_ver]
    cid_matrix_aug = [cid_matrix_flip_hor, cid_matrix_flip_ver, cid_matrix_flip_hor_ver]
    visit_dist_aug = [visit_dist_flip_hor, visit_dist_flip_ver, visit_dist_flip_hor_ver]
    return cid_matrix_aug, visit_dist_aug

def switch(cid_matrix, next_turn, outcome):
    cid_matrix_black_mask = cid_matrix > 7
    cid_matrix_red_mask = (cid_matrix > 0) * (1 - cid_matrix_black_mask)
    cid_matrix_red = cid_matrix_black_mask * cid_matrix - 7 * cid_matrix_black_mask
    cid_matrix_black = cid_matrix_red_mask * cid_matrix + 7 * cid_matrix_red_mask
    return cid_matrix_red + cid_matrix_black, 1 - next_turn, -outcome


class XQDataset(Dataset):
    def __init__(self, cid_matrices, next_turns, outcomes, visit_dists, visit_update_mask, aug=False):
        super().__init__()
        self.cid_matrices = copy.deepcopy(cid_matrices)
        self.next_turns = copy.deepcopy(next_turns)
        self.outcomes = copy.deepcopy(outcomes)
        self.visit_dists = copy.deepcopy(visit_dists)
        self.visit_update_mask = copy.deepcopy(visit_update_mask)
        if aug:
            cid_matrices_aug = []
            next_turns_aug = []
            outcomes_aug = []
            visit_dists_aug = []
            visit_update_mask_aug = []
            for i, m in enumerate(cid_matrices):
                cid_matrices_flip_aug, visits_flip_aug = flip(m, visit_dists[i])
                cid_matrices_aug.extend(cid_matrices_flip_aug)
                visit_dists_aug.extend(visits_flip_aug)
                visit_update_mask_aug.extend([visit_update_mask[i]] * 3)
                next_turns_aug.extend([next_turns[i]] * 3)
                outcomes_aug.extend([outcomes[i]] * 3)
                cid_matrix_switch, next_turn_switch, outcome_switch = switch(m, next_turns[i], outcomes[i])
                cid_matrices_aug.append(cid_matrix_switch)
                next_turns_aug.append(next_turn_switch)
                outcomes_aug.append(outcome_switch)
                visit_dists_aug.append(visit_dists[i])
                visit_update_mask_aug.append(visit_update_mask[i])
                cid_matrix_switch_flips, visit_dist_switch_flips = flip(cid_matrix_switch, visit_dists[i])
                cid_matrices_aug.extend(cid_matrix_switch_flips)
                next_turns_aug.extend([next_turn_switch] * 3)
                outcomes_aug.extend([outcome_switch] * 3)
                visit_dists_aug.extend(visit_dist_switch_flips)
                visit_update_mask_aug.extend([visit_update_mask[i]] * 3)

            self.cid_matrices.extend(cid_matrices_aug)
            self.next_turns.extend(next_turns_aug)
            self.outcomes.extend(outcomes_aug)
            self.visit_dists.extend(visit_dists_aug)
            self.visit_update_mask.extend(visit_update_mask_aug)

    def __getitem__(self, index):
        return self.cid_matrices[index], self.next_turns[index], self.outcomes[index], self.visit_dists[index], self.visit_update_mask[index]

    def __len__(self):
        return len(self.cid_matrices)

if __name__ == '__main__':
    import pickle
    import copy
    import random
    from board import Board
    pkl = 'data/buffer.pkl'
    with open(pkl, 'rb') as f:
        cid_matrices_buffer, next_turns_buffer, outcomes_buffer, visit_dists_buffer, visit_update_mask_buffer = pickle.load(f)
    # cnt = 0
    # for cid_matrix, next_turn, outcome, visit_dist, visit_update_mask in zip(cid_matrices_buffer, next_turns_buffer, outcomes_buffer, visit_dists_buffer, visit_update_mask_buffer):
    #     cnt += 1
    #     print(np.sum(visit_dist), cnt)
    #     if cnt >= 3300:
    #         break
    # exit(0)
    for cid_matrix, next_turn, outcome, visit_dist, visit_update_mask in zip(cid_matrices_buffer, next_turns_buffer, outcomes_buffer, visit_dists_buffer, visit_update_mask_buffer):
        board_ = Board()
        print(f'Original sample:')
        board_.load_state(cid_matrix, next_turn)
        board_.show_board()

        cid_matrix_s, next_turn_s, outcome_s = switch(cid_matrix, next_turn, outcome)
        board_.load_state(cid_matrix_s, next_turn_s)
        print(f'Switched sample:')
        board_.show_board()

        # break
        visit_dist[19 * 90 + 21] = 1
        cid_matrix_aug, visit_dist_aug = flip(cid_matrix, visit_dist)
        board_ = Board()
        # board_.move(9, 2, 7, 4)
        print(f'Original sample:')
        board_.load_state(cid_matrix, next_turn)
        actions = []
        action_dist = []
        for j, (src_row, src_col) in enumerate(board_.feasible_srcs):
            src_idx = src_row * 9 + src_col
            for dst_row, dst_col in board_.feasible_dsts[j]:
                dst_idx = dst_row * 9 + dst_col
                action_idx = src_idx * 90 + dst_idx
                action_dist.append(visit_dist[action_idx])
                actions.append((src_row, src_col, dst_row, dst_col))
        a = np.argmax(action_dist)
        action = actions[a]
        board_.move(*action)
        board_.show_board()
        print(f'Performing action: from ({action[0]}, {action[1]}) to ({action[2]}, {action[3]}), with prob {action_dist[a]:.4f}')
        
        for i, (cid_matrix_a, visit_dist_a) in enumerate(zip(cid_matrix_aug, visit_dist_aug)):
            print(f'Augmented sample {i + 1}:')
            board_.load_state(cid_matrix_a, next_turn)
            actions = []
            action_dist = []
            for j, (src_row, src_col) in enumerate(board_.feasible_srcs):
                src_idx = src_row * 9 + src_col
                for dst_row, dst_col in board_.feasible_dsts[j]:
                    dst_idx = dst_row * 9 + dst_col
                    action_idx = src_idx * 90 + dst_idx
                    action_dist.append(visit_dist_a[action_idx])
                    actions.append((src_row, src_col, dst_row, dst_col))
            a = np.argmax(action_dist)
            action = actions[a]
            print(f'Performing action: from ({action[0]}, {action[1]}) to ({action[2]}, {action[3]}), with prob {action_dist[a]:.4f}')
            board_.move(*action)
            board_.show_board()
        break


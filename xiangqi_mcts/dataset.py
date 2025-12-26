from torch.utils.data import Dataset
import actions
import numpy as np

def flip_visits_hori(visit_dist):
    visit_dist_flip_hor = np.zeros_like(visit_dist)
    for i in range(192):
        pid, row_delta, col_delta = actions.index2move[i]
        flip_idx = actions.move2index[(pid, row_delta, -col_delta)]
        visit_dist_flip_hor[flip_idx] = visit_dist[i]
    
    return visit_dist_flip_hor


def flip_visits_vert(visit_dist):
    visit_dist_flip_ver = np.zeros_like(visit_dist)
    for i in range(192):
        pid, row_delta, col_delta = actions.index2move[i]
        flip_idx = actions.move2index[(pid, -row_delta, col_delta)]
        visit_dist_flip_ver[flip_idx] = visit_dist[i]
    return visit_dist_flip_ver


class XQDataset(Dataset):
    def __init__(self, coords, next_turns, isdeads, outcomes, visit_dists, visit_update_mask, steps, aug=False):
        super().__init__()
        self.coords = list(coords)
        self.next_turns = list(next_turns)
        self.outcomes = list(outcomes)
        self.isdeads = list(isdeads)
        self.steps = list(steps)
        self.visit_dists = list(visit_dists)
        self.visit_update_mask = list(visit_update_mask)
        if aug:
            coords_aug = []
            next_turns_aug = []
            outcomes_aug = []
            isdeads_aug = []
            steps_aug = []
            visit_dists_aug = []
            visit_update_mask_aug = []
            for i, coords_board in enumerate(coords):
                # flip horizontally
                coords_flip_hori = coords_board.copy()
                coords_flip_hori[:, 1] = 8 - coords_flip_hori[:, 1]
                visit_dists_flip_hori = flip_visits_hori(visit_dists[i])
                coords_aug.append(coords_flip_hori)
                visit_dists_aug.append(visit_dists_flip_hori)
                visit_update_mask_aug.append(visit_update_mask[i])
                isdeads_aug.append(isdeads[i])
                steps_aug.append(steps[i])
                next_turns_aug.append(next_turns[i])
                outcomes_aug.append(outcomes[i])
                # flip vertically
                coords_flip_ver = coords_board.copy()
                coords_flip_ver[:, 0] = 9 - coords_flip_ver[:, 0]
                visit_dists_flip_ver = flip_visits_vert(visit_dists[i])
                coords_aug.append(coords_flip_ver)
                visit_dists_aug.append(visit_dists_flip_ver)
                visit_update_mask_aug.append(visit_update_mask[i])
                next_turns_aug.append(next_turns[i])
                outcomes_aug.append(outcomes[i])
                isdeads_aug.append(isdeads[i])
                steps_aug.append(steps[i])
                # flip both
                coords_flip_both = coords_flip_hori.copy()
                coords_flip_both[:, 0] = 9 - coords_flip_both[:, 0]
                visit_dists_flip_both = flip_visits_vert(visit_dists_flip_hori)
                coords_aug.append(coords_flip_both)
                visit_dists_aug.append(visit_dists_flip_both)
                visit_update_mask_aug.append(visit_update_mask[i])
                next_turns_aug.append(next_turns[i])
                outcomes_aug.append(outcomes[i])
                isdeads_aug.append(isdeads[i])
                steps_aug.append(steps[i])
                # switch sides
                coords_switch = coords_board.copy()
                coords_switch[:16] = coords_board[16:]
                coords_switch[16:] = coords_board[:16]
                coords_aug.append(coords_switch)
                visit_dists_aug.append(visit_dists[i])
                visit_update_mask_aug.append(visit_update_mask[i])
                next_turns_aug.append(1 - next_turns[i])
                outcomes_aug.append(-outcomes[i])
                isdeads_switch = isdeads[i].copy()
                isdeads_switch[:16], isdeads_switch[16:] = isdeads[i][16:], isdeads[i][:16]
                isdeads_aug.append(isdeads_switch)
                steps_aug.append(steps[i])

                # switch sides + flip horizontally
                coords_switch_hori = coords_switch.copy()
                coords_switch_hori[:, 1] = 8 - coords_switch_hori[:, 1]
                visit_dists_switch_hori = flip_visits_hori(visit_dists[i])
                coords_aug.append(coords_switch_hori)
                visit_dists_aug.append(visit_dists_switch_hori)
                visit_update_mask_aug.append(visit_update_mask[i])
                next_turns_aug.append(1 - next_turns[i])
                outcomes_aug.append(-outcomes[i])
                isdeads_aug.append(isdeads_switch)
                steps_aug.append(steps[i])
                # switch sides + flip vertically
                coords_switch_ver = coords_switch.copy()
                coords_switch_ver[:, 0] = 9 - coords_switch_ver[:, 0]
                visit_dists_switch_ver = flip_visits_vert(visit_dists[i])
                coords_aug.append(coords_switch_ver)
                visit_dists_aug.append(visit_dists_switch_ver)
                visit_update_mask_aug.append(visit_update_mask[i])
                next_turns_aug.append(1 - next_turns[i])
                outcomes_aug.append(-outcomes[i])
                isdeads_aug.append(isdeads_switch)
                steps_aug.append(steps[i])
                # switch sides + flip both
                coords_switch_both = coords_switch_hori.copy()
                coords_switch_both[:, 0] = 9 - coords_switch_both[:, 0]
                visit_dists_switch_both = flip_visits_vert(visit_dists_switch_hori)
                coords_aug.append(coords_switch_both)
                visit_dists_aug.append(visit_dists_switch_both)
                visit_update_mask_aug.append(visit_update_mask[i])
                next_turns_aug.append(1 - next_turns[i])
                outcomes_aug.append(-outcomes[i])
                isdeads_aug.append(isdeads_switch)
                steps_aug.append(steps[i])



            self.coords.extend(coords_aug)
            self.next_turns.extend(next_turns_aug)
            self.outcomes.extend(outcomes_aug)
            self.isdeads.extend(isdeads_aug)
            self.steps.extend(steps_aug)
            self.visit_dists.extend(visit_dists_aug)
            self.visit_update_mask.extend(visit_update_mask_aug)

    def __getitem__(self, index):
        return self.coords[index], self.next_turns[index], self.outcomes[index], self.visit_dists[index], self.visit_update_mask[index], self.isdeads[index], self.steps[index]

    def __len__(self):
        return len(self.coords)


if __name__ == '__main__':
    import pickle
    import copy
    import random
    from board import Board
    pkl = 'data/buffer.pkl'
    with open(pkl, 'rb') as f:
        coords_buffer, next_turns_buffer, isdeads_buffer, outcomes_buffer, visits_buffer, visit_update_mask_buffer, steps_buffer = pickle.load(f)
    
    idx = random.choice(range(100))

    i = 0
    for coords, next_turn, isdeads, outcome, visit_dist, visit_update_mask, steps in zip(coords_buffer, next_turns_buffer, isdeads_buffer, outcomes_buffer, visits_buffer, visit_update_mask_buffer, steps_buffer):
        if i != idx:
            i += 1
            continue

        act_idx = random.choice(range(192))
        visit_dist[act_idx] = 1
        print('original act idx', act_idx)
        board_ = Board()
        print(f'Original sample:')
        board_.load_state(coords, next_turn, isdeads)
        board_.show_board()

        coords_switch = coords.copy()
        coords_switch[:16] = coords[16:]
        coords_switch[16:] = coords[:16]
        board_.load_state(coords_switch, 1 - next_turn, isdeads)
        pid, row_delta, col_delta = actions.index2move[act_idx]
        src_row, src_col = board_.pieces[pid].row, board_.pieces[pid].col
        dst_row = src_row + row_delta
        dst_col = src_col + col_delta
        print(f'switched. moving from {pid} {(src_row, src_col)} to {(dst_row, dst_col)}')
        board_.show_board()

        coords_switch_hori = coords_switch.copy()
        coords_switch_hori[:, 1] = 8 - coords_switch_hori[:, 1]
        visit_dists_switch_hori = flip_visits_hori(visit_dist)
        board_.load_state(coords_switch_hori, 1 - next_turn, isdeads)
        act_idx = np.argmax(visit_dists_switch_hori)
        print('filpped act idx', act_idx)
        pid, row_delta, col_delta = actions.index2move[act_idx]
        src_row, src_col = board_.pieces[pid].row, board_.pieces[pid].col
        dst_row = src_row + row_delta
        dst_col = src_col + col_delta
        print(f'switched and flip hori. moving from {pid} {(src_row, src_col)} to {(dst_row, dst_col)}')
        board_.show_board()
        break

        '''
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
        '''


import numpy as np

from piece import Piece
import state
from config import piece_cid_to_name, color_id_to_str

NROW = 10
NCOL = 9


class Board:
    def __init__(self, next_turn='red', state_str=None, my_color='red'):
        self.pieces = []
        self.my_color = my_color
        self.next_turn = next_turn
        if state_str is not None:
            self.load_state(state_str)
        else:
            self.init_board()
        self.feasible_srcs, self.feasible_dsts = self.get_feasible_moves()

    def init_board(self):
        board = [[None for j in range(NCOL)] for i in range(NROW)]
        if self.my_color == 'red':
            row_ju_black = 0
            row_ju_red = 9
            row_pao_black = 2
            row_pao_red = 7
            row_zu_black = 3
            row_zu_red = 6
        else:
            row_ju_black = 9
            row_ju_red = 0
            row_pao_black = 7
            row_pao_red = 2
            row_zu_black = 6
            row_zu_red = 3
        board[row_ju_black][0] = Piece('black', 'ju')
        board[row_ju_black][1] = Piece('black', 'ma')
        board[row_ju_black][2] = Piece('black', 'xiang')
        board[row_ju_black][3] = Piece('black', 'shi')
        board[row_ju_black][4] = Piece('black', 'king')
        board[row_ju_black][5] = Piece('black', 'shi')
        board[row_ju_black][6] = Piece('black', 'xiang')
        board[row_ju_black][7] = Piece('black', 'ma')
        board[row_ju_black][8] = Piece('black', 'ju')

        board[row_ju_red][0] = Piece('red', 'ju')
        board[row_ju_red][1] = Piece('red', 'ma')
        board[row_ju_red][2] = Piece('red', 'xiang')
        board[row_ju_red][3] = Piece('red', 'shi')
        board[row_ju_red][4] = Piece('red', 'king')
        board[row_ju_red][5] = Piece('red', 'shi')
        board[row_ju_red][6] = Piece('red', 'xiang')
        board[row_ju_red][7] = Piece('red', 'ma')
        board[row_ju_red][8] = Piece('red', 'ju')

        board[row_pao_black][1] = Piece('black', 'pao')
        board[row_pao_black][7] = Piece('black', 'pao')

        board[row_pao_red][1] = Piece('red', 'pao')
        board[row_pao_red][7] = Piece('red', 'pao')

        board[row_zu_black][0] = Piece('black', 'zu')
        board[row_zu_black][2] = Piece('black', 'zu')
        board[row_zu_black][4] = Piece('black', 'zu')
        board[row_zu_black][6] = Piece('black', 'zu')
        board[row_zu_black][8] = Piece('black', 'zu')

        board[row_zu_red][0] = Piece('red', 'zu')
        board[row_zu_red][2] = Piece('red', 'zu')
        board[row_zu_red][4] = Piece('red', 'zu')
        board[row_zu_red][6] = Piece('red', 'zu')
        board[row_zu_red][8] = Piece('red', 'zu')

        for i in range(NROW):
            for j in range(NCOL):
                if board[i][j] is None:
                    board[i][j] = Piece('none', 'none')
        
        self.board = board
    
        self.last_move = None
        self.step = 0


    def get_cid_matrix(self):
        cid_matrix = np.zeros([10, 9], dtype=int)
        for i in range(NROW):
            for j in range(NCOL):
                cid_matrix[i, j] = self.board[i][j].get_cid()
        return cid_matrix


    def shift_turn(self):
        if self.next_turn == 'red':
            self.next_turn = 'black'
        else:
            self.next_turn = 'red'

    def load_state(self, state_str):
        split = state_str.split('|')
        cids = split[:-3]
        self.next_turn = split[-3]
        self.step = int(split[-2])
        self.my_color = split[-1]

        for i in range(NROW):
            for j in range(NCOL):
                cid = int(cids[i * NCOL + j])
                if cid == 0:
                    color = 'none'
                elif cid > 7:
                    color = 'black'
                    cid = cid - 7
                else:
                    color = 'red'
                category = piece_cid_to_name[cid]
                piece = Piece(color, category)
                self.board[i][j] = piece

    def dump_state(self):
        # board state_str format: color|cid|next_turn|last_move
        cid_matrix = self.get_cid_matrix()
        cid_matrix_ = [str(n) for n in cid_matrix.reshape(-1).tolist()]
        cid_matrix_str = '|'.join(cid_matrix_)
        state_str = cid_matrix_str + '|' + self.next_turn + '|' + \
            str(self.step) + '|' + self.my_color
        return state_str

    def show_board(self, src_row=None, src_col=None, dst_row=None, dst_col=None):
        show = ''
        context = '\x1b[6;30;42m'
        context_end = '\x1b[0m'
        CRED = '\033[91m'
        CEND = '\033[0m'
        for i in range(NROW):
            for j in range(NCOL):
                piece = self.board[i][j]
                if piece.category == 'none':
                    if src_row is not None and i == src_row and j == src_col:
                        show += f'{context}　{context_end}'
                    else:
                        # show += '　'
                        show += '\033[33m十\033[0m'
                else:
                    if piece.color == 'red':
                        char = f'{CRED}{piece.get_char()}{CEND}'
                    else:
                        char = piece.get_char()
                    if src_row is not None and i == dst_row and j == dst_col:
                        show += f'{context}{char}{context_end}'
                    else:
                        show += char

            show += '\n'
        print(show)

    def check_king_facing(self, src_pos, dst_pos):
        dst_row, dst_col = dst_pos
        src_row, src_col = src_pos
        
        removed = self.move(src_row, src_col, dst_row, dst_col, leave_feasible=True)
        if removed.category == 'king':
            self.restore(removed, src_row, src_col, dst_row, dst_col, leave_feasible=True)
            return False

        king1_row, king1_col = -1, -1
        for i in range(3):
            for j in range(3, 6):
                if self.board[i][j].category == 'king':
                    king1_row, king1_col = i, j
                    break
        
        king2_row, king2_col = -1, -1
        for i in range(7, 10):
            for j in range(3, 6):
                if self.board[i][j].category == 'king':
                    king2_row, king2_col = i, j
                    break
        
        if king1_col != king2_col:
            self.restore(removed, src_row, src_col, dst_row, dst_col, leave_feasible=True)
            return False

        for i in range(king1_row + 1, king2_row):
            if self.board[i][king1_col].category != 'none':
                self.restore(removed, src_row, src_col, dst_row, dst_col, leave_feasible=True)
                return False
        
        self.restore(removed, src_row, src_col, dst_row, dst_col, leave_feasible=True)
        return True

    def get_feasible_moves(self):
        all_destinies = []
        all_sources = []
        for i in range(NROW):
            for j in range(NCOL):
                piece = self.board[i][j]
                if piece.category == 'none':
                    continue
                if piece.color != self.next_turn:
                    continue
                piece_destinies = []
                if piece.category == 'ju':
                    # search left
                    for c in range(j - 1, -1, -1):
                        piece_ = self.board[i][c]
                        if piece_.category == 'none':
                            piece_destinies.append((i, c))
                        elif piece_.color != self.next_turn:
                            piece_destinies.append((i, c))
                            break
                        else:
                            break

                    # search right
                    for c in range(j + 1, NCOL):
                        piece_ = self.board[i][c]
                        if piece_.category == 'none':
                            piece_destinies.append((i, c))
                        elif piece_.color != self.next_turn:
                            piece_destinies.append((i, c))
                            break
                        else:
                            break

                    # search upper
                    for r in range(i - 1, -1, -1):
                        piece_ = self.board[r][j]
                        if piece_.category == 'none':
                            piece_destinies.append((r, j))
                        elif piece_.color != self.next_turn:
                            piece_destinies.append((r, j))
                            break
                        else:
                            break

                    # search lower
                    for r in range(i + 1, NROW):
                        piece_ = self.board[r][j]
                        if piece_.category == 'none':
                            piece_destinies.append((r, j))
                        elif piece_.color != self.next_turn:
                            piece_destinies.append((r, j))
                            break
                        else:
                            break

                elif piece.category == 'ma':
                    # search left
                    if j > 1 and self.board[i][j - 1].category == 'none':  # 不别马腿
                        # left upper
                        if i > 0:
                            dst_row, dst_col = i - 1, j - 2
                            if self.board[dst_row][dst_col].color != piece.color:
                                piece_destinies.append((dst_row, dst_col))
                        # left below
                        if i < NROW - 1:
                            dst_row, dst_col = i + 1, j - 2
                            if self.board[dst_row][dst_col].color != piece.color:
                                piece_destinies.append((dst_row, dst_col))
                    # search right
                    if j < NCOL - 2 and self.board[i][j + 1].category == 'none':
                        # right upper
                        if i > 0:
                            dst_row, dst_col = i - 1, j + 2
                            if self.board[dst_row][dst_col].color != piece.color:
                                piece_destinies.append((dst_row, dst_col))
                        # left below
                        if i < NROW - 1:
                            dst_row, dst_col = i + 1, j + 2
                            if self.board[dst_row][dst_col].color != piece.color:
                                piece_destinies.append((dst_row, dst_col))
                    # search upper
                    if i > 1 and self.board[i - 1][j].category == 'none':
                        # upper left
                        if j > 0:
                            dst_row, dst_col = i - 2, j - 1
                            if self.board[dst_row][dst_col].color != piece.color:
                                piece_destinies.append((dst_row, dst_col))
                        # upper right
                        if j < NCOL - 1:
                            dst_row, dst_col = i - 2, j + 1
                            if self.board[dst_row][dst_col].color != piece.color:
                                piece_destinies.append((dst_row, dst_col))
                    # search below
                    if i < NROW - 2 and self.board[i + 1][j].category == 'none':
                        # below left
                        if j > 0:
                            dst_row, dst_col = i + 2, j - 1
                            if self.board[dst_row][dst_col].color != piece.color:
                                piece_destinies.append((dst_row, dst_col))
                        # below right
                        if j < NCOL - 1:
                            dst_row, dst_col = i + 2, j + 1
                            if self.board[dst_row][dst_col].color != piece.color:
                                piece_destinies.append((dst_row, dst_col))

                elif piece.category == 'xiang':
                    if j > 1:
                        # left upper
                        if (piece.color != self.my_color and i > 1) or (
                                piece.color == self.my_color and i > 5):
                            if self.board[i - 1][j - 1].category == 'none':
                                dst_row, dst_col = i - 2, j - 2
                                if self.board[dst_row][dst_col].color != piece.color:
                                    piece_destinies.append((dst_row, dst_col))
                        # left lower
                        if (piece.color != self.my_color and i < 3) or (
                                piece.color == self.my_color and i < 8):
                            if self.board[i + 1][j - 1].category == 'none':
                                dst_row, dst_col = i + 2, j - 2
                                if self.board[dst_row][dst_col].color != piece.color:
                                    piece_destinies.append((dst_row, dst_col))
                    if j < 7:
                        # right upper
                        if (piece.color != self.my_color and i > 1) or (
                                piece.color == self.my_color and i > 5):
                            if self.board[i - 1][j + 1].category == 'none':
                                dst_row, dst_col = i - 2, j + 2
                                if self.board[dst_row][dst_col].color != piece.color:
                                    piece_destinies.append((dst_row, dst_col))
                        # right lower
                        if (piece.color != self.my_color and i < 3) or (
                                piece.color == self.my_color and i < 8):
                            if self.board[i + 1][j + 1].category == 'none':
                                dst_row, dst_col = i + 2, j + 2
                                if self.board[dst_row][dst_col].color != piece.color:
                                    piece_destinies.append((dst_row, dst_col))

                elif piece.category == 'shi':
                    # upper left
                    if (j == 3 and i == 0) or (j == 3 and i == 7):
                        dst_row, dst_col = i + 1, j + 1
                        if self.board[dst_row][dst_col].color != piece.color:
                            piece_destinies.append((dst_row, dst_col))
                    # upper right
                    if (j == 5 and i == 0) or (j == 5 and i == 7):
                        dst_row, dst_col = i + 1, j - 1
                        if self.board[dst_row][dst_col].color != piece.color:
                            piece_destinies.append((dst_row, dst_col))
                    # lower left
                    if (j == 3 and i == 2) or (j == 3 and i == 9):
                        dst_row, dst_col = i - 1, j + 1
                        if self.board[dst_row][dst_col].color != piece.color:
                            piece_destinies.append((dst_row, dst_col))
                    # lower right
                    if (j == 5 and i == 2) or (j == 5 and i == 9):
                        dst_row, dst_col = i - 1, j - 1
                        if self.board[dst_row][dst_col].color != piece.color:
                            piece_destinies.append((dst_row, dst_col))
                    # middle
                    if (j == 4 and i == 1) or (j == 4 and i == 8):
                        pos_ = [(i - 1, j - 1), (i - 1, j + 1),
                                (i + 1, j - 1), (i + 1, j + 1)]
                        for pos in pos_:
                            if self.board[pos[0]][pos[1]].color != piece.color:
                                piece_destinies.append(pos)

                elif piece.category == 'king':
                    # left
                    if j > 3:
                        dst_row, dst_col = i, j - 1
                        if self.board[dst_row][dst_col].color != piece.color:
                            piece_destinies.append((dst_row, dst_col))
                    # right
                    if j < 5:
                        dst_row, dst_col = i, j + 1
                        if self.board[dst_row][dst_col].color != piece.color:
                            piece_destinies.append((dst_row, dst_col))
                    # upper
                    if (i > 0 and piece.color != self.my_color) or (i > 7 and piece.color == self.my_color):
                        dst_row, dst_col = i - 1, j
                        if self.board[dst_row][dst_col].color != piece.color:
                            piece_destinies.append((dst_row, dst_col))
                    # lower
                    if (i < 2 and piece.color != self.my_color) or (i < 9 and piece.color == self.my_color):
                        dst_row, dst_col = i + 1, j
                        if self.board[dst_row][dst_col].color != piece.color:
                            piece_destinies.append((dst_row, dst_col))

                elif piece.category == 'pao':
                    # search left
                    for c in range(j - 1, -1, -1):
                        piece_ = self.board[i][c]
                        if piece_.category == 'none':
                            piece_destinies.append((i, c))
                        else:
                            if c > 0:
                                for d in range(c - 1, -1, -1):
                                    piece__ = self.board[i][d]
                                    if piece__.category == 'none':
                                        continue
                                    else:
                                        if piece__.color != piece.color:
                                            piece_destinies.append((i, d))
                                        break
                            break

                    # search right
                    for c in range(j + 1, NCOL):
                        piece_ = self.board[i][c]
                        if piece_.category == 'none':
                            piece_destinies.append((i, c))
                        else:
                            if c < 8:
                                for d in range(c + 1, NCOL):
                                    piece__ = self.board[i][d]
                                    if piece__.category == 'none':
                                        continue
                                    else:
                                        if piece__.color != piece.color:
                                            piece_destinies.append((i, d))
                                        break
                            break

                    # search upper
                    for r in range(i - 1, -1, -1):
                        piece_ = self.board[r][j]
                        if piece_.category == 'none':
                            piece_destinies.append((r, j))
                        else:
                            if r > 0:
                                for e in range(r - 1, -1, -1):
                                    piece__ = self.board[e][j]
                                    if piece__.category == 'none':
                                        continue
                                    else:
                                        if piece__.color != piece.color:
                                            piece_destinies.append((e, j))
                                        break
                            break
                    # search lower
                    for r in range(i + 1, NROW):
                        piece_ = self.board[r][j]
                        if piece_.category == 'none':
                            piece_destinies.append((r, j))
                        else:
                            if r < 8:
                                for e in range(r + 1, NROW):
                                    piece__ = self.board[e][j]
                                    if piece__.category == 'none':
                                        continue
                                    else:
                                        if piece__.color != piece.color:
                                            piece_destinies.append((e, j))
                                        break
                            break

                else: # 兵卒
                    pos_ = []
                    if piece.color != self.my_color:
                        if i < 9:
                            pos = (i + 1, j)
                            pos_.append(pos)
                        if i > 4:
                            if j > 0:
                                pos_.append((i, j - 1))
                            if j < 8:
                                pos_.append((i, j + 1))
                    if piece.color == self.my_color:
                        if i > 0:
                            pos = (i - 1, j)
                            pos_.append(pos)
                        if i < 5:
                            if j > 0:
                                pos_.append((i, j - 1))
                            if j < 8:
                                pos_.append((i, j + 1))
                    for pos in pos_:
                        if self.board[pos[0]][pos[1]].color != piece.color:
                            piece_destinies.append(pos)
                piece_destinies = [dst_pos for dst_pos in piece_destinies if not self.check_king_facing((i, j), dst_pos)]
                all_destinies.append(piece_destinies)
                all_sources.append((i, j))

        all_sources = [p for i, p in enumerate(all_sources) if len(all_destinies[i]) > 0]
        all_destinies = [p for p in all_destinies if len(p) > 0]

        return all_sources, all_destinies

    def move(self, src_row, src_col, dst_row, dst_col, leave_feasible=False):
        src_piece = self.board[src_row][src_col]
        dst_piece = self.board[dst_row][dst_col]
        self.board[dst_row][dst_col] = src_piece
        self.board[src_row][src_col] = Piece('none', 'none')
        removed = dst_piece
        self.shift_turn()
        if not leave_feasible:
            self.feasible_srcs, self.feasible_dsts = self.get_feasible_moves()
        self.step += 1
        return removed 

    def restore(self, removed_piece, src_row, src_col, dst_row, dst_col, leave_feasible=False):
        moved_piece = self.board[dst_row][dst_col]
        self.board[src_row][src_col] = moved_piece
        self.board[dst_row][dst_col] = removed_piece
        self.shift_turn()
        self.step -= 1
        if not leave_feasible:
            self.feasible_srcs, self.feasible_dsts = self.get_feasible_moves()
            

    def get_result(self):
        king_is_dead = True
        for i in range(3):
            for j in range(3, 6):
                if self.board[i][j].category == 'king':
                    king_is_dead = False
                    break
        if king_is_dead:
            if self.my_color == 'red':
                return 'red'
            else:
                return 'black'
        
        king_is_dead = True
        for i in range(7, 10):
            for j in range(3, 6):
                if self.board[i][j].category == 'king':
                    king_is_dead = False
                    break
        if king_is_dead:
            if self.my_color == 'red':
                return 'black'
            else:
                return 'red'

        num_move = 0
        for moves_p in self.feasible_dsts:
            num_move += len(moves_p)
        if num_move == 0:
            return 'red' if self.next_turn == 'black' else 'black'
        if self.step >= 100:
            return 'draw'
        return 'going'


if __name__ == '__main__':
    import random
    board = Board(my_color='black')
    board.show_board()
    for i in range(10):
        print(f'========{board.step}==========')
        srcs, dsts = board.get_feasible_moves()
        src_idx = random.choice(range(len(srcs)))
        src_row, src_col = srcs[src_idx]
        dst_row, dst_col = random.choice(dsts[src_idx])
        board.move(src_row, src_col, dst_row, dst_col)
        board.show_board(src_row, src_col, dst_row, dst_col)

    

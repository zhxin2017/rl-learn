import numpy as np

from piece import Piece
import state
from config import piece_cid_to_name, color_id_to_str

NROW = 10
NCOL = 9


class Board:
    def __init__(self, next_turn='red', state_str=None, me_color='red'):
        self.pos_to_piece = {}
        self.pieces = []
        self.me_color = me_color
        self.next_turn = next_turn
        if state_str is not None:
            self.load_state(state_str)
        else:
            self.init_board()

    def init_board(self):
        self.cid_matrix = np.zeros([10, 9], dtype=int)
        self.color_matrix = np.ones([10, 9], dtype=int) * 2
        self.last_move = None
        self.step = 0
        self.played_result = None
        for r in range(NROW):
            for c in range(NCOL):
                self.pos_to_piece[(r, c)] = None
        if self.me_color == 'red':
            row0_black, row2_black, row3_black = 0, 2, 3
            row0_red, row2_red, row3_red = 9, 7, 6
        else:
            row0_black, row2_black, row3_black = 9, 7, 6
            row0_red, row2_red, row3_red = 0, 2, 3

        ju1black = Piece(row0_black, 0, 'black', 'ju')
        ma1black = Piece(row0_black, 1, 'black', 'ma')
        xiang1black = Piece(row0_black, 2, 'black', 'xiang')
        shi1black = Piece(row0_black, 3, 'black', 'shi')
        king_black = Piece(row0_black, 4, 'black', 'king')
        shi2black = Piece(row0_black, 5, 'black', 'shi')
        xiang2black = Piece(row0_black, 6, 'black', 'xiang')
        ma2black = Piece(row0_black, 7, 'black', 'ma')
        ju2black = Piece(row0_black, 8, 'black', 'ju')
        pao1black = Piece(row2_black, 1, 'black', 'pao')
        pao2black = Piece(row2_black, 7, 'black', 'pao')
        zu1black = Piece(row3_black, 0, 'black', 'zu')
        zu2black = Piece(row3_black, 2, 'black', 'zu')
        zu3black = Piece(row3_black, 4, 'black', 'zu')
        zu4black = Piece(row3_black, 6, 'black', 'zu')
        zu5black = Piece(row3_black, 8, 'black', 'zu')

        ju1red = Piece(row0_red, 0, 'red', 'ju')
        ma1red = Piece(row0_red, 1, 'red', 'ma')
        xiang1red = Piece(row0_red, 2, 'red', 'xiang')
        shi1red = Piece(row0_red, 3, 'red', 'shi')
        king_red = Piece(row0_red, 4, 'red', 'king')
        shi2red = Piece(row0_red, 5, 'red', 'shi')
        xiang2red = Piece(row0_red, 6, 'red', 'xiang')
        ma2red = Piece(row0_red, 7, 'red', 'ma')
        ju2red = Piece(row0_red, 8, 'red', 'ju')
        pao1red = Piece(row2_red, 1, 'red', 'pao')
        pao2red = Piece(row2_red, 7, 'red', 'pao')
        zu1red = Piece(row3_red, 0, 'red', 'zu')
        zu2red = Piece(row3_red, 2, 'red', 'zu')
        zu3red = Piece(row3_red, 4, 'red', 'zu')
        zu4red = Piece(row3_red, 6, 'red', 'zu')
        zu5red = Piece(row3_red, 8, 'red', 'zu')

        self.king_red = king_red
        self.king_black = king_black

        self.pieces = [king_red, king_black, ju1red, ma1red, xiang1red, shi1red, shi2red, xiang2red, ma2red,
                       ju2red, pao1red, pao2red, zu1red, zu2red, zu3red, zu4red, zu5red,
                       ju1black, ma1black, xiang1black, shi1black, shi2black, xiang2black, ma2black,
                       ju2black, pao1black, pao2black, zu1black, zu2black, zu3black, zu4black, zu5black]
        for piece in self.pieces:
            self.pos_to_piece[(piece.row, piece.col)] = piece
            self.color_matrix[piece.row, piece.col] = piece.get_color_id()
            self.cid_matrix[piece.row, piece.col] = piece.get_cid()

    def shift_turn(self):
        if self.next_turn == 'red':
            self.next_turn = 'black'
        else:
            self.next_turn = 'red'

    def load_state(self, state_str):
        cid_matrix, color_matrix, next_turn, last_move, step, result = state.parse_state_str_for_board(state_str)
        self.next_turn = next_turn
        self.last_move = last_move
        self.cid_matrix = cid_matrix
        self.color_matrix = color_matrix
        self.step = step
        self.played_result = result

        self.pieces = []
        self.king_black = None
        self.king_red = None
        for i in range(NROW):
            for j in range(NCOL):
                if cid_matrix[i, j] == 0:
                    self.pos_to_piece[(i, j)] = None
                    continue
                piece = Piece(i, j,
                              color_id_to_str[color_matrix[i, j].item()],
                              piece_cid_to_name[cid_matrix[i, j].item()])
                self.pos_to_piece[(i, j)] = piece
                if cid_matrix[i, j] == 5:
                    if color_id_to_str[color_matrix[i, j].item()] == 'red':
                        self.king_red = piece
                    else:
                        self.king_black = piece
                self.pieces.append(piece)
        if self.king_red is not None:
            if self.king_red.row < 3:
                self.me_color = 'black'
            else:
                self.me_color = 'red'
        else:
            if self.king_black.row < 3:
                self.me_color = 'red'
            else:
                self.me_color = 'black'

    def dump_state(self):
        # board state_str format: color|cid|next_turn|last_move
        return state.dump_state_str_from_board(self.color_matrix, self.cid_matrix, self.last_move, self.next_turn)

    def show_board(self):
        show = ''
        context = '\x1b[6;30;42m'
        context_end = '\x1b[0m'
        CRED = '\033[91m'
        CEND = '\033[0m'
        if self.last_move is None:
            src_row = src_col = dst_row = dst_col = None
        else:
            src_row, src_col, dst_row, dst_col = self.last_move
        for i in range(NROW):
            for j in range(NCOL):
                piece = self.pos_to_piece[(i, j)]
                if piece is None:
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

    def check_king_facing(self, moving_piece, dst_col):
        if moving_piece.category != 'king':
            if self.king_red.col != self.king_black.col:
                return False
            if moving_piece.col != self.king_red.col:
                return False
            if moving_piece.col == dst_col:
                return False

            min_row = min(self.king_red.row, self.king_black.row)
            max_row = max(self.king_red.row, self.king_black.row)
            n_blockade = 0
            for i in range(min_row + 1, max_row):
                if self.pos_to_piece[(i, self.king_red.col)] is not None:
                    n_blockade += 1
                    if n_blockade > 1:
                        return False
            return True
        else:
            if self.king_red.col == self.king_black.col:
                return False
            if (moving_piece.color == 'red' and dst_col == self.king_black.col or
                    moving_piece.color == 'black' and dst_col == self.king_red.col):
                min_row = min(self.king_red.row, self.king_black.row)
                max_row = max(self.king_red.row, self.king_black.row)
                for i in range(min_row + 1, max_row):
                    if self.pos_to_piece[(i, dst_col)] is not None:
                        return False
                return True

    def feasible_moves(self):
        pieces = [p for p in self.pieces if p.color == self.next_turn]
        all_destinies = []
        for piece in pieces:
            piece_destinies = []
            if piece.category == 'ju':
                # search left
                for c in range(piece.col - 1, -1, -1):
                    piece_ = self.pos_to_piece.get((piece.row, c))
                    if piece_ is None:
                        piece_destinies.append((piece.row, c))
                    elif piece_.color != self.next_turn:
                        piece_destinies.append((piece.row, c))
                        break
                    else:
                        break

                # search right
                for c in range(piece.col + 1, NCOL):
                    piece_ = self.pos_to_piece.get((piece.row, c))
                    if piece_ is None:
                        piece_destinies.append((piece.row, c))
                    elif piece_.color != self.next_turn:
                        piece_destinies.append((piece.row, c))
                        break
                    else:
                        break

                # search upper
                for r in range(piece.row - 1, -1, -1):
                    piece_ = self.pos_to_piece.get((r, piece.col))
                    if piece_ is None:
                        piece_destinies.append((r, piece.col))
                    elif piece_.color != self.next_turn:
                        piece_destinies.append((r, piece.col))
                        break
                    else:
                        break

                # search lower
                for r in range(piece.row + 1, NROW):
                    piece_ = self.pos_to_piece.get((r, piece.col))
                    if piece_ is None:
                        piece_destinies.append((r, piece.col))
                    elif piece_.color != self.next_turn:
                        piece_destinies.append((r, piece.col))
                        break
                    else:
                        break

            elif piece.category == 'ma':
                # search left
                if piece.col > 1:
                    if self.pos_to_piece[(piece.row, piece.col - 1)] is None:  # 不别马腿
                        # left upper
                        if piece.row > 0:
                            pos = (piece.row - 1, piece.col - 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                piece_destinies.append(pos)
                        # left below
                        if piece.row < NROW - 1:
                            pos = (piece.row + 1, piece.col - 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                piece_destinies.append(pos)
                # search right
                if piece.col < NCOL - 2:
                    if self.pos_to_piece[(piece.row, piece.col + 1)] is None:
                        # right upper
                        if piece.row > 0:
                            pos = (piece.row - 1, piece.col + 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                piece_destinies.append(pos)
                        # left below
                        if piece.row < NROW - 1:
                            pos = (piece.row + 1, piece.col + 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                piece_destinies.append(pos)
                # search upper
                if piece.row > 1:
                    if self.pos_to_piece[(piece.row - 1, piece.col)] is None:
                        # upper left
                        if piece.col > 0:
                            pos = (piece.row - 2, piece.col - 1)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                piece_destinies.append(pos)
                        # upper right
                        if piece.col < NCOL - 1:
                            pos = (piece.row - 2, piece.col + 1)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                piece_destinies.append(pos)
                # search below
                if piece.row < NROW - 2:
                    if self.pos_to_piece[(piece.row + 1, piece.col)] is None:
                        # below left
                        if piece.col > 0:
                            pos = (piece.row + 2, piece.col - 1)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                piece_destinies.append(pos)
                        # below right
                        if piece.col < NCOL - 1:
                            pos = (piece.row + 2, piece.col + 1)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                piece_destinies.append(pos)

            elif piece.category == 'xiang':

                if piece.col > 1:
                    # left upper
                    if (piece.color != self.me_color and piece.row > 1) or (
                            piece.color == self.me_color and piece.row > 5):
                        if self.pos_to_piece[(piece.row - 1, piece.col - 1)] is None:
                            pos = (piece.row - 2, piece.col - 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                piece_destinies.append(pos)
                    # left lower
                    if (piece.color != self.me_color and piece.row < 3) or (
                            piece.color == self.me_color and piece.row < 8):
                        if self.pos_to_piece[(piece.row + 1, piece.col - 1)] is None:
                            pos = (piece.row + 2, piece.col - 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                piece_destinies.append(pos)
                if piece.col < 7:
                    # right upper
                    if (piece.color != self.me_color and piece.row > 1) or (
                            piece.color == self.me_color and piece.row > 5):
                        if self.pos_to_piece[(piece.row - 1, piece.col + 1)] is None:
                            pos = (piece.row - 2, piece.col + 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                piece_destinies.append(pos)
                    # right lower
                    if (piece.color != self.me_color and piece.row < 3) or (
                            piece.color == self.me_color and piece.row < 8):
                        if self.pos_to_piece[(piece.row + 1, piece.col + 1)] is None:
                            pos = (piece.row + 2, piece.col + 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                piece_destinies.append(pos)

            elif piece.category == 'shi':
                # upper left
                if (piece.col == 3 and piece.row == 0) or (piece.col == 3 and piece.row == 7):
                    pos = (piece.row + 1, piece.col + 1)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                        piece_destinies.append(pos)
                # upper right
                if (piece.col == 5 and piece.row == 0) or (piece.col == 5 and piece.row == 7):
                    pos = (piece.row + 1, piece.col - 1)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                        piece_destinies.append(pos)
                # lower left
                if (piece.col == 3 and piece.row == 2) or (piece.col == 3 and piece.row == 9):
                    pos = (piece.row - 1, piece.col + 1)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                        piece_destinies.append(pos)
                # lower right
                if (piece.col == 5 and piece.row == 2) or (piece.col == 5 and piece.row == 9):
                    pos = (piece.row - 1, piece.col - 1)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                        piece_destinies.append(pos)
                # middle
                if (piece.col == 4 and piece.row == 1) or (piece.col == 4 and piece.row == 8):
                    pos_ = [(piece.row - 1, piece.col - 1), (piece.row - 1, piece.col + 1),
                            (piece.row + 1, piece.col - 1), (piece.row + 1, piece.col + 1)]
                    for pos in pos_:
                        if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                            piece_destinies.append(pos)

            elif piece.category == 'king':
                # left
                if piece.col > 3:
                    pos = (piece.row, piece.col - 1)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                        piece_destinies.append(pos)
                # right
                if piece.col < 5:
                    pos = (piece.row, piece.col + 1)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                        piece_destinies.append(pos)
                # upper
                if (piece.row > 0 and piece.color != self.me_color) or (piece.row > 7 and piece.color == self.me_color):
                    pos = (piece.row - 1, piece.col)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                        piece_destinies.append(pos)
                # lower
                if (piece.row < 2 and piece.color != self.me_color) or (piece.row < 9 and piece.color == self.me_color):
                    pos = (piece.row + 1, piece.col)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                        piece_destinies.append(pos)

            elif piece.category == 'pao':
                # search left
                for c in range(piece.col - 1, -1, -1):
                    piece_ = self.pos_to_piece.get((piece.row, c))
                    if piece_ is None:
                        piece_destinies.append((piece.row, c))
                    else:
                        if c > 0:
                            for d in range(c - 1, -1, -1):
                                pos = (piece.row, d)
                                piece__ = self.pos_to_piece[pos]
                                if piece__ is None:
                                    continue
                                if piece__.color != piece.color:
                                    piece_destinies.append(pos)
                                    break
                        break

                # search right
                for c in range(piece.col + 1, NCOL):
                    piece_ = self.pos_to_piece.get((piece.row, c))
                    if piece_ is None:
                        piece_destinies.append((piece.row, c))
                    else:
                        if c < 8:
                            for d in range(c + 1, NCOL):
                                pos = (piece.row, d)
                                piece__ = self.pos_to_piece[pos]
                                if piece__ is None:
                                    continue
                                if piece__.color != piece.color:
                                    piece_destinies.append(pos)
                                    break
                        break

                # search upper
                for r in range(piece.row - 1, -1, -1):
                    piece_ = self.pos_to_piece.get((r, piece.col))
                    if piece_ is None:
                        piece_destinies.append((r, piece.col))
                    else:
                        if r > 0:
                            for e in range(r - 1, -1, -1):
                                pos = (e, piece.col)
                                piece__ = self.pos_to_piece[pos]
                                if piece__ is None:
                                    continue
                                if piece__.color != piece.color:
                                    piece_destinies.append(pos)
                                    break
                        break
                # search lower
                for r in range(piece.row + 1, NROW):
                    piece_ = self.pos_to_piece.get((r, piece.col))
                    if piece_ is None:
                        piece_destinies.append((r, piece.col))
                    else:
                        if r < 8:
                            for e in range(r + 1, NROW):
                                pos = (e, piece.col)
                                piece__ = self.pos_to_piece[pos]
                                if piece__ is None:
                                    continue
                                if piece__.color != piece.color:
                                    piece_destinies.append(pos)
                                    break
                        break

            else:
                pos_ = []
                # black
                if piece.color != self.me_color:
                    if piece.row < 9:
                        pos = (piece.row + 1, piece.col)
                        pos_.append(pos)
                    if piece.row > 4:
                        if piece.col > 0:
                            pos_.append((piece.row, piece.col - 1))
                        if piece.col < 8:
                            pos_.append((piece.row, piece.col + 1))
                # red
                if piece.color == self.me_color:
                    if piece.row > 0:
                        pos = (piece.row - 1, piece.col)
                        pos_.append(pos)
                    if piece.row < 5:
                        if piece.col > 0:
                            pos_.append((piece.row, piece.col - 1))
                        if piece.col < 8:
                            pos_.append((piece.row, piece.col + 1))
                for pos in pos_:
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                        piece_destinies.append(pos)
            piece_destinies = [dst_pos for dst_pos in piece_destinies if not self.check_king_facing(piece, dst_pos[1])]
            all_destinies.append(piece_destinies)

        pieces = [pc for i, pc in enumerate(pieces) if len(all_destinies[i]) > 0]
        all_destinies = [p for p in all_destinies if len(p) > 0]

        return pieces, all_destinies

    def move(self, src_row, src_col, dst_row, dst_col):
        src_piece = self.pos_to_piece[(src_row, src_col)]
        dst_piece = self.pos_to_piece[(dst_row, dst_col)]
        self.color_matrix[dst_row, dst_col] = src_piece.get_color_id()
        self.cid_matrix[dst_row, dst_col] = src_piece.get_cid()
        self.color_matrix[src_row, src_col] = 2
        self.cid_matrix[src_row, src_col] = 0
        if dst_piece is not None:
            dst_index = self.pieces.index(dst_piece)
            removed = self.pieces.pop(dst_index)
            if removed.category == 'king':
                if removed.color == 'red':
                    self.king_red = None
                else:
                    self.king_black = None
        else:
            removed = None
        self.pos_to_piece[(src_row, src_col)] = None
        self.pos_to_piece[(dst_row, dst_col)] = src_piece
        src_piece.row = dst_row
        src_piece.col = dst_col
        self.shift_turn()
        prev_last_move = self.last_move
        self.last_move = (src_row, src_col, dst_row, dst_col)
        self.step += 1
        return removed, prev_last_move

    def restore(self, removed_piece, prev_last_move):
        src_row, src_col, dst_row, dst_col = self.last_move
        moved_piece = self.pos_to_piece[(dst_row, dst_col)]
        self.pos_to_piece[(src_row, src_col)] = moved_piece
        moved_piece.row = src_row
        moved_piece.col = src_col
        self.cid_matrix[src_row, src_col] = moved_piece.get_cid()
        self.color_matrix[src_row, src_col] = moved_piece.get_color_id()

        self.pos_to_piece[(dst_row, dst_col)] = removed_piece
        if removed_piece is not None:
            self.cid_matrix[dst_row, dst_col] = removed_piece.get_cid()
            self.color_matrix[dst_row, dst_col] = removed_piece.get_color_id()
            self.pieces.append(removed_piece)
        else:
            self.cid_matrix[dst_row, dst_col] = 0
            self.color_matrix[dst_row, dst_col] = 2

        self.shift_turn()
        self.step -= 1
        self.last_move = prev_last_move

    def get_result(self):
        if self.king_black is None:
            return 'red'
        if self.king_red is None:
            return 'black'
        _, moves = self.feasible_moves()
        num_move = 0
        for moves_p in moves:
            num_move += len(moves_p)
        if num_move == 0:
            return 'red' if self.next_turn == 'black' else 'black'
        return 'going'


if __name__ == '__main__':
    board = Board(me_color='black')
    removed, prev_last_move = board.move(0, 0, 0, 4)
    board.show_board()
    print(board.get_result())
    board.restore(removed, prev_last_move)
    board.show_board()
    print(board.get_result())

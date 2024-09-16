import numpy as np
from dataset import parse_state
from config import color_str_to_id, piece_name_to_id

nrow = 10
ncol = 9

piece_id_to_name = {v: k for k, v in piece_name_to_id.items()}
color_id_to_str = {v: k for k, v in color_str_to_id.items()}

class Piece:
    def __init__(self, row, col, color, category):
        self.row = row
        self.col = col
        self.color = color
        self.category = category

    def get_cid(self):
        return piece_name_to_id[self.category]

    def get_char(self):
        char_dict = {
            'red': {
                'ju': '俥',
                'ma': '傌',
                'xiang': '相',
                'shi': '仕',
                'jiang': '帥',
                'pao': '炮',
                'zu': '兵'
            },
            'black': {
                'ju': '車',
                'ma': '馬',
                'xiang': '象',
                'shi': '士',
                'jiang': '将',
                'pao': '砲',
                'zu': '卒'
            }
        }
        return char_dict[self.color][self.category]


class Board:
    def __init__(self, next_turn='red', state_str=None):
        self.pos_to_piece = {}
        if state_str is not None:
            self.load_state(state_str)
        else:
            self.init_board(next_turn)

    def init_board(self, first_turn):
        self.board_matrix = np.zeros([10, 9], dtype=int)
        self.next_turn = first_turn
        for r in range(nrow):
            for c in range(ncol):
                self.pos_to_piece[(r, c)] = None
        ju1black = Piece(0, 0, 'black', 'ju')
        ma1black = Piece(0, 1, 'black', 'ma')
        xiang1black = Piece(0, 2, 'black', 'xiang')
        shi1black = Piece(0, 3, 'black', 'shi')
        jiangblack = Piece(0, 4, 'black', 'jiang')
        shi2black = Piece(0, 5, 'black', 'shi')
        xiang2black = Piece(0, 6, 'black', 'xiang')
        ma2black = Piece(0, 7, 'black', 'ma')
        ju2black = Piece(0, 8, 'black', 'ju')
        pao1black = Piece(2, 1, 'black', 'pao')
        pao2black = Piece(2, 7, 'black', 'pao')
        zu1black = Piece(3, 0, 'black', 'zu')
        zu2black = Piece(3, 2, 'black', 'zu')
        zu3black = Piece(3, 4, 'black', 'zu')
        zu4black = Piece(3, 6, 'black', 'zu')
        zu5black = Piece(3, 8, 'black', 'zu')

        ju1red = Piece(9, 0, 'red', 'ju')
        ma1red = Piece(9, 1, 'red', 'ma')
        xiang1red = Piece(9, 2, 'red', 'xiang')
        shi1red = Piece(9, 3, 'red', 'shi')
        jiangred = Piece(9, 4, 'red', 'jiang')
        shi2red = Piece(9, 5, 'red', 'shi')
        xiang2red = Piece(9, 6, 'red', 'xiang')
        ma2red = Piece(9, 7, 'red', 'ma')
        ju2red = Piece(9, 8, 'red', 'ju')
        pao1red = Piece(7, 1, 'red', 'pao')
        pao2red = Piece(7, 7, 'red', 'pao')
        zu1red = Piece(6, 0, 'red', 'zu')
        zu2red = Piece(6, 2, 'red', 'zu')
        zu3red = Piece(6, 4, 'red', 'zu')
        zu4red = Piece(6, 6, 'red', 'zu')
        zu5red = Piece(6, 8, 'red', 'zu')

        self.jiangred = jiangred
        self.jiangblack = jiangblack

        self.pieces = [jiangred, jiangblack, ju1red, ma1red, xiang1red, shi1red, shi2red, xiang2red, ma2red,
                       ju2red, pao1red, pao2red, zu1red, zu2red, zu3red, zu4red, zu5red,
                       ju1black, ma1black, xiang1black, shi1black, shi2black, xiang2black, ma2black,
                       ju2black, pao1black, pao2black, zu1black, zu2black, zu3black, zu4black, zu5black]
        for piece in self.pieces:
            self.board_matrix[piece.row, piece.col] = color_str_to_id[piece.color] * 10 + piece.get_cid()

        self.pos_to_piece[(9, 0)] = ju1red
        self.pos_to_piece[(9, 1)] = ma1red
        self.pos_to_piece[(9, 2)] = xiang1red
        self.pos_to_piece[(9, 3)] = shi1red
        self.pos_to_piece[(9, 4)] = jiangred
        self.pos_to_piece[(9, 5)] = shi2red
        self.pos_to_piece[(9, 6)] = xiang2red
        self.pos_to_piece[(9, 7)] = ma2red
        self.pos_to_piece[(9, 8)] = ju2red
        self.pos_to_piece[(7, 1)] = pao1red
        self.pos_to_piece[(7, 7)] = pao2red
        self.pos_to_piece[(6, 0)] = zu1red
        self.pos_to_piece[(6, 2)] = zu2red
        self.pos_to_piece[(6, 4)] = zu3red
        self.pos_to_piece[(6, 6)] = zu4red
        self.pos_to_piece[(6, 8)] = zu5red

        self.pos_to_piece[(0, 0)] = ju1black
        self.pos_to_piece[(0, 1)] = ma1black
        self.pos_to_piece[(0, 2)] = xiang1black
        self.pos_to_piece[(0, 3)] = shi1black
        self.pos_to_piece[(0, 4)] = jiangblack
        self.pos_to_piece[(0, 5)] = shi2black
        self.pos_to_piece[(0, 6)] = xiang2black
        self.pos_to_piece[(0, 7)] = ma2black
        self.pos_to_piece[(0, 8)] = ju2black
        self.pos_to_piece[(2, 1)] = pao1black
        self.pos_to_piece[(2, 7)] = pao2black
        self.pos_to_piece[(3, 0)] = zu1black
        self.pos_to_piece[(3, 2)] = zu2black
        self.pos_to_piece[(3, 4)] = zu3black
        self.pos_to_piece[(3, 6)] = zu4black
        self.pos_to_piece[(3, 8)] = zu5black

    def gen_formatted_state(self):
        board_ = self.board_matrix.reshape(-1)
        state = ''.join(['0' * (1 - b // 10) + str(b) for b in board_]) + '|' + self.next_turn
        return state

    def load_state(self, state_str):
        cid, color, next_turn, board_matrix = parse_state(state_str)
        self.next_turn = next_turn
        self.board_matrix = board_matrix

        self.pieces = []
        self.jiangblack = None
        self.jiangred = None
        for i in range(nrow):
            for j in range(ncol):
                if cid[i, j] < 1:
                    self.pos_to_piece[(i, j)] = None
                    continue
                piece = Piece(i, j, color_id_to_str[color[i, j].item()], piece_id_to_name[cid[i, j].item()])
                self.pos_to_piece[(i, j)] = piece
                if cid[i, j] == 5:
                    if color_id_to_str[color[i, j]] == 'red':
                        self.jiangred = piece
                    else:
                        self.jiangblack = piece
                self.pieces.append(piece)

    def show_board(self, src_row=None, src_col=None, dst_row=None, dst_col=None):
        show = ''
        context = '\x1b[6;30;42m'
        context_end = '\x1b[0m'
        CRED = '\033[91m'
        CEND = '\033[0m'
        for i in range(nrow):
            for j in range(ncol):
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

    def check_jiang_facing(self, moving_piece, dst_pos):
        if moving_piece.category != 'jiang':
            if self.jiangred.col != self.jiangblack.col:
                return False
            if moving_piece.col != self.jiangred.col:
                return False
            if moving_piece.col == dst_pos[1]:
                return False
            n_blockade = 0
            for i in range(self.jiangblack.row + 1, self.jiangred.row):
                if self.pos_to_piece[(i, self.jiangred.col)] is not None:
                    n_blockade += 1
                    if n_blockade > 1:
                        return False
            return True
        else:
            if self.jiangred.col == self.jiangblack.col:
                return False
            if moving_piece.col == dst_pos[1]:
                return False
            row_red = self.jiangred.row
            row_black = self.jiangblack.row
            if moving_piece.color == 0:
                col_red = dst_pos[1]
                col_black = self.jiangblack.col
            else:
                col_red = self.jiangred.col
                col_black = dst_pos[1]
            if col_red != col_black:
                return False
            for i in range(row_black + 1, row_red):
                if self.pos_to_piece[(i, col_red)] is not None:
                    return False
            return True

    def feasible_moves(self):
        pieces = [p for p in self.pieces if p.color == self.next_turn]
        destinies = []
        for piece in pieces:
            destinies_ = []
            if piece.category == 'ju':
                # search left
                for c in range(piece.col - 1, -1, -1):
                    piece_ = self.pos_to_piece.get((piece.row, c))
                    if piece_ is None:
                        destinies_.append((piece.row, c))
                    elif piece_.color != self.next_turn:
                        destinies_.append((piece.row, c))
                        break
                    else:
                        break

                # search right
                for c in range(piece.col + 1, ncol):
                    piece_ = self.pos_to_piece.get((piece.row, c))
                    if piece_ is None:
                        destinies_.append((piece.row, c))
                    elif piece_.color != self.next_turn:
                        destinies_.append((piece.row, c))
                        break
                    else:
                        break

                # search upper
                for r in range(piece.row - 1, -1, -1):
                    piece_ = self.pos_to_piece.get((r, piece.col))
                    if piece_ is None:
                        destinies_.append((r, piece.col))
                    elif piece_.color != self.next_turn:
                        destinies_.append((r, piece.col))
                        break
                    else:
                        break

                # search lower
                for r in range(piece.row + 1, nrow):
                    piece_ = self.pos_to_piece.get((r, piece.col))
                    if piece_ is None:
                        destinies_.append((r, piece.col))
                    elif piece_.color != self.next_turn:
                        destinies_.append((r, piece.col))
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
                                destinies_.append(pos)
                        # left below
                        if piece.row < nrow - 1:
                            pos = (piece.row + 1, piece.col - 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                destinies_.append(pos)
                # search right
                if piece.col < ncol - 2:
                    if self.pos_to_piece[(piece.row, piece.col + 1)] is None:
                        # right upper
                        if piece.row > 0:
                            pos = (piece.row - 1, piece.col + 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                destinies_.append(pos)
                        # left below
                        if piece.row < nrow - 1:
                            pos = (piece.row + 1, piece.col + 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                destinies_.append(pos)
                # search upper
                if piece.row > 1:
                    if self.pos_to_piece[(piece.row - 1, piece.col)] is None:
                        # upper left
                        if piece.col > 0:
                            pos = (piece.row - 2, piece.col - 1)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                destinies_.append(pos)
                        # upper right
                        if piece.col < ncol - 1:
                            pos = (piece.row - 2, piece.col + 1)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                destinies_.append(pos)
                # search below
                if piece.row < nrow - 2:
                    if self.pos_to_piece[(piece.row + 1, piece.col)] is None:
                        # below left
                        if piece.col > 0:
                            pos = (piece.row + 2, piece.col - 1)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                destinies_.append(pos)
                        # below right
                        if piece.col < ncol - 1:
                            pos = (piece.row + 2, piece.col + 1)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                destinies_.append(pos)

            elif piece.category == 'xiang':
                if piece.col > 1:
                    # left upper
                    if (piece.color == 1 and piece.row > 1) or (piece.color == 0 and piece.row > 5):
                        if self.pos_to_piece[(piece.row - 1, piece.col - 1)] is None:
                            pos = (piece.row - 2, piece.col - 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                destinies_.append(pos)
                    # left lower
                    if (piece.color == 1 and piece.row < 3) or (piece.color == 0 and piece.row < 8):
                        if self.pos_to_piece[(piece.row + 1, piece.col - 1)] is None:
                            pos = (piece.row + 2, piece.col - 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                destinies_.append(pos)
                if piece.col < 7:
                    # right upper
                    if (piece.color == 1 and piece.row > 1) or (piece.color == 0 and piece.row > 5):
                        if self.pos_to_piece[(piece.row - 1, piece.col + 1)] is None:
                            pos = (piece.row - 2, piece.col + 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                destinies_.append(pos)
                    # right lower
                    if (piece.color == 1 and piece.row < 3) or (piece.color == 0 and piece.row < 8):
                        if self.pos_to_piece[(piece.row + 1, piece.col + 1)] is None:
                            pos = (piece.row + 2, piece.col + 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                                destinies_.append(pos)

            elif piece.category == 'shi':
                # upper left
                if (piece.col == 3 and piece.row == 0) or (piece.col == 3 and piece.row == 7):
                    pos = (piece.row + 1, piece.col + 1)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                        destinies_.append(pos)
                # upper right
                if (piece.col == 5 and piece.row == 0) or (piece.col == 5 and piece.row == 7):
                    pos = (piece.row + 1, piece.col - 1)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                        destinies_.append(pos)
                # lower left
                if (piece.col == 3 and piece.row == 2) or (piece.col == 3 and piece.row == 9):
                    pos = (piece.row - 1, piece.col + 1)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                        destinies_.append(pos)
                # lower right
                if (piece.col == 5 and piece.row == 2) or (piece.col == 5 and piece.row == 9):
                    pos = (piece.row - 1, piece.col - 1)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                        destinies_.append(pos)
                # middle
                if (piece.col == 4 and piece.row == 1) or (piece.col == 4 and piece.row == 8):
                    pos_ = [(piece.row - 1, piece.col - 1), (piece.row - 1, piece.col + 1),
                            (piece.row + 1, piece.col - 1), (piece.row + 1, piece.col + 1)]
                    for pos in pos_:
                        if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                            destinies_.append(pos)

            elif piece.category == 'jiang':
                # left
                if piece.col > 3:
                    pos = (piece.row, piece.col - 1)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                        destinies_.append(pos)
                # right
                if piece.col < 5:
                    pos = (piece.row, piece.col + 1)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                        destinies_.append(pos)
                # upper
                if (piece.row > 0 and piece.color == 1) or (piece.row > 7 and piece.color == 0):
                    pos = (piece.row - 1, piece.col)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                        destinies_.append(pos)
                # lower
                if (piece.row < 0 and piece.color == 1) or (piece.row < 9 and piece.color == 0):
                    pos = (piece.row + 1, piece.col)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color != piece.color:
                        destinies_.append(pos)

            elif piece.category == 'pao':
                # search left
                for c in range(piece.col - 1, -1, -1):
                    piece_ = self.pos_to_piece.get((piece.row, c))
                    if piece_ is None:
                        destinies_.append((piece.row, c))
                    else:
                        if c > 0:
                            for d in range(c - 1, -1, -1):
                                pos = (piece.row, d)
                                piece__ = self.pos_to_piece[pos]
                                if piece__ is None:
                                    continue
                                if piece__.color != piece.color:
                                    destinies_.append(pos)
                                    break
                        break

                # search right
                for c in range(piece.col + 1, ncol):
                    piece_ = self.pos_to_piece.get((piece.row, c))
                    if piece_ is None:
                        destinies_.append((piece.row, c))
                    else:
                        if c < 8:
                            for d in range(c + 1, ncol):
                                pos = (piece.row, d)
                                piece__ = self.pos_to_piece[pos]
                                if piece__ is None:
                                    continue
                                if piece__.color != piece.color:
                                    destinies_.append(pos)
                                    break
                        break

                # search upper
                for r in range(piece.row - 1, -1, -1):
                    piece_ = self.pos_to_piece.get((r, piece.col))
                    if piece_ is None:
                        destinies_.append((r, piece.col))
                    else:
                        if r > 0:
                            for e in range(r - 1, -1, -1):
                                pos = (e, piece.col)
                                piece__ = self.pos_to_piece[pos]
                                if piece__ is None:
                                    continue
                                if piece__.color != piece.color:
                                    destinies_.append(pos)
                                    break
                        break
                # search lower
                for r in range(piece.row + 1, nrow):
                    piece_ = self.pos_to_piece.get((r, piece.col))
                    if piece_ is None:
                        destinies_.append((r, piece.col))
                    else:
                        if r < 8:
                            for e in range(r + 1, nrow):
                                pos = (e, piece.col)
                                piece__ = self.pos_to_piece[pos]
                                if piece__ is None:
                                    continue
                                if piece__.color != piece.color:
                                    destinies_.append(pos)
                                    break
                        break

            else:
                pos_ = []
                # black
                if piece.color == 1:
                    if piece.row < 9:
                        pos = (piece.row + 1, piece.col)
                        pos_.append(pos)
                    if piece.row > 4:
                        if piece.col > 0:
                            pos_.append((piece.row, piece.col - 1))
                        if piece.col < 8:
                            pos_.append((piece.row, piece.col + 1))
                # red
                if piece.color == 0:
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
                        destinies_.append(pos)
            destinies_ = [p for p in destinies_ if not self.check_jiang_facing(piece, p)]
            destinies.append(destinies_)

        pieces = [pc for i, pc in enumerate(pieces) if len(destinies[i]) > 0]
        destinies = [p for p in destinies if len(p) > 0]

        return pieces, destinies

    def move(self, src_row, src_col, dst_row, dst_col):
        src_piece = self.pos_to_piece[(src_row, src_col)]
        dst_piece = self.pos_to_piece[(dst_row, dst_col)]
        if dst_piece is not None:
            dst_index = self.pieces.index(dst_piece)
            removed = self.pieces.pop(dst_index)
        else:
            removed = None
        self.pos_to_piece[(src_row, src_col)] = None
        self.pos_to_piece[(dst_row, dst_col)] = src_piece
        self.board_matrix[dst_row, dst_col] = color_str_to_id[src_piece.color] * 10 + src_piece.get_cid()
        self.board_matrix[src_row, src_col] = 0
        src_piece.row = dst_row
        src_piece.col = dst_col
        self.next_turn = 'black' if self.next_turn == 'red' else 'red'
        return removed

    def restore(self, src_row, src_col, dst_row, dst_col, removed_piece):
        moved_piece = self.pos_to_piece[(dst_row, dst_col)]
        self.pos_to_piece[(src_row, src_col)] = moved_piece
        self.board_matrix[src_row, src_col] = color_str_to_id[moved_piece.color] * 10 + moved_piece.get_cid()
        moved_piece.row = src_row
        moved_piece.col = src_col

        self.pos_to_piece[(dst_row, dst_col)] = removed_piece
        if removed_piece is not None:
            self.board_matrix[dst_row, dst_col] = color_str_to_id[removed_piece.color] * 10 + removed_piece.get_cid()
            self.pieces.append(removed_piece)
        else:
            self.board_matrix[dst_row, dst_col] = 0

        self.next_turn = 'black' if self.next_turn == 'red' else 'red'

    def get_result(self):
        if self.next_turn == 'black' and (self.jiangblack is None or self.jiangblack not in self.pieces):
            return 'red'
        if self.next_turn == 'red' and (self.jiangred is None or self.jiangred not in self.pieces):
            return 'black'
        _, moves = self.feasible_moves()
        num_move = 0
        for moves_p in moves:
            num_move += len(moves_p)
        if num_move == 0:
            return 'red' if self.next_turn == 'black' else 'black'
        return 'draw'


if __name__ == '__main__':
    board = Board()
    board.move(7, 1, 0, 1)
    board.show_board(7, 1, 0, 1)
    print(len(board.pieces))
    pieces, moves = board.feasible_moves()
    print(moves)

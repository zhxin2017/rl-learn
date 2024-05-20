import numpy as np


nrow = 10
ncol = 9

color = {0: 'red', 1: 'black'}

class Piece:
    def __init__(self, row, col, color, category):
        self.row = row
        self.col = col
        self.color = color
        self.category = category

    def get_cid(self):
        cid_dict = {
            'ju':1,
            'ma':2,
            'xiang':3,
            'shi':4,
            'jiang':5,
            'pao':6,
            'zu':7
        }
        return cid_dict[self.category]


class Board:
    def __init__(self, first_turn=0):
        self.board_matrix = np.zeros([10, 9], dtype=int)
        self.pos_to_piece = {}
        self.next_turn = first_turn
        for r in range(nrow):
            for c in range(ncol):
                self.pos_to_piece[(r, c)] = None
        ju1black = Piece(0, 0, 1, 'ju')
        ma1black = Piece(0, 1, 1, 'ma')
        xiang1black = Piece(0, 2, 1, 'xiang')
        shi1black= Piece(0, 3, 1, 'shi')
        jiangblack = Piece(0, 4, 1, 'jiang')
        shi2black= Piece(0, 5, 1, 'shi')
        xiang2black = Piece(0, 6, 1, 'xiang')
        ma2black = Piece(0, 7, 1, 'ma')
        ju2black = Piece(0, 8, 1, 'ju')
        pao1black = Piece(2, 1, 1, 'pao')
        pao2black = Piece(2, 7, 1, 'pao')
        zu1black = Piece(3, 0, 1, 'zu')
        zu2black = Piece(3, 2, 1, 'zu')
        zu3black = Piece(3, 4, 1, 'zu')
        zu4black = Piece(3, 6, 1, 'zu')
        zu5black = Piece(3, 8, 1, 'zu')

        ju1red = Piece(9, 0, 0, 'ju')
        ma1red = Piece(9, 1, 0, 'ma')
        xiang1red = Piece(9, 2, 0, 'xiang')
        shi1red= Piece(9, 3, 0, 'shi')
        jiangred = Piece(9, 4, 0, 'jiang')
        shi2red= Piece(9, 5, 0, 'shi')
        xiang2red = Piece(9, 6, 0, 'xiang')
        ma2red = Piece(9, 7, 0, 'ma')
        ju2red = Piece(9, 8, 0, 'ju')
        pao1red = Piece(7, 1, 0, 'pao')
        pao2red = Piece(7, 7, 0, 'pao')
        zu1red = Piece(6, 0, 0, 'zu')
        zu2red = Piece(6, 2, 0, 'zu')
        zu3red = Piece(6, 4, 0, 'zu')
        zu4red = Piece(6, 6, 0, 'zu')
        zu5red = Piece(6, 8, 0, 'zu')

        self.jiangred = jiangred
        self.jiangblack = jiangblack

        self.pieces = [jiangred, jiangblack, ju1red, ma1red, xiang1red, shi1red, shi2red, xiang2red, ma2red,
                       ju2red, pao1red, pao2red, zu1red, zu2red, zu3red, zu4red, zu5red,
                       ju1black, ma1black, xiang1black, shi1black, shi2black, xiang2black, ma2black,
                       ju2black, pao1black, pao2black, zu1black, zu2black, zu3black, zu4black, zu5black]
        for piece in self.pieces:
            self.board_matrix[piece.row, piece.col] = piece.color * 10 + piece.get_cid()

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

    def get_state(self):
        board_ = self.board_matrix.reshape(-1)
        state = ''.join(['0' * (1 - b // 10) + str(b) for b in board_]) + '|' + str(self.next_turn)
        return state

    def check_jiang_facing(self, moving_piece, dst_pos):
        jiang_red, jiang_black = self.pieces[:2]
        if moving_piece.category != 'jiang':
            if jiang_red.col != jiang_black.col:
                return False
            if moving_piece.col != jiang_red.col:
                return False
            if moving_piece.col == dst_pos[1]:
                return False
            n_blockade = 0
            for i in range(jiang_black.row + 1, jiang_red.row):
                if self.pos_to_piece[(i, jiang_red.col)] is not None:
                    n_blockade += 1
                    if n_blockade > 1:
                        return False
            return True
        else:
            if jiang_red.col == jiang_black.col:
                return False
            if moving_piece.col == dst_pos[1]:
                return False
            row_red = jiang_red.row
            row_black = jiang_black.row
            if moving_piece.color == 0:
                col_red = dst_pos[1]
                col_black = jiang_black.col
            else:
                col_red = jiang_red.col
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
                    elif piece_.color == 1 - self.next_turn:
                        destinies_.append((piece.row, c))
                        break
                    else:
                        break

                # search right
                for c in range(piece.col + 1, ncol):
                    piece_ = self.pos_to_piece.get((piece.row, c))
                    if piece_ is None:
                        destinies_.append((piece.row, c))
                    elif piece_.color == 1 - self.next_turn:
                        destinies_.append((piece.row, c))
                        break
                    else:
                        break

                # search upper
                for r in range(piece.row - 1, -1, -1):
                    piece_ = self.pos_to_piece.get((r, piece.col))
                    if piece_ is None:
                        destinies_.append((r, piece.col))
                    elif piece_.color == 1 - self.next_turn:
                        destinies_.append((r, piece.col))
                        break
                    else:
                        break

                # search lower
                for r in range(piece.row + 1, nrow):
                    piece_ = self.pos_to_piece.get((r, piece.col))
                    if piece_ is None:
                        destinies_.append((r, piece.col))
                    elif piece_.color == 1 - self.next_turn:
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
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
                                destinies_.append(pos)
                        # left below
                        if piece.row < nrow - 1:
                            pos = (piece.row + 1, piece.col - 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
                                destinies_.append(pos)
                # search right
                if piece.col < ncol - 2:
                    if self.pos_to_piece[(piece.row, piece.col + 1)] is None:
                        # right upper
                        if piece.row > 0:
                            pos = (piece.row - 1, piece.col + 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
                                destinies_.append(pos)
                        # left below
                        if piece.row < nrow - 1:
                            pos = (piece.row + 1, piece.col + 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
                                destinies_.append(pos)
                # search upper
                if piece.row > 1:
                    if self.pos_to_piece[(piece.row - 1, piece.col)] is None:
                        # upper left
                        if piece.col > 0:
                            pos = (piece.row - 2, piece.col - 1)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
                                destinies_.append(pos)
                        # upper right
                        if piece.col < ncol - 1:
                            pos = (piece.row - 2, piece.col + 1)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
                                destinies_.append(pos)
                # search below
                if piece.row < nrow - 2:
                    if self.pos_to_piece[(piece.row + 1, piece.col)] is None:
                        # below left
                        if piece.col > 0:
                            pos = (piece.row + 2, piece.col - 1)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
                                destinies_.append(pos)
                        # below right
                        if piece.col < ncol - 1:
                            pos = (piece.row + 2, piece.col + 1)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
                                destinies_.append(pos)

            elif piece.category == 'xiang':
                if piece.col > 1:
                    # left upper
                    if (piece.color == 1 and piece.row > 1) or (piece.color == 0 and piece.row > 5):
                        if self.pos_to_piece[(piece.row - 1, piece.col - 1)] is None:
                            pos = (piece.row - 2, piece.col - 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
                                destinies_.append(pos)
                    # left lower
                    if (piece.color == 1 and piece.row < 3) or (piece.color == 0 and piece.row < 8):
                        if self.pos_to_piece[(piece.row + 1, piece.col - 1)] is None:
                            pos = (piece.row + 2, piece.col - 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
                                destinies_.append(pos)
                if piece.col < 7:
                    # right upper
                    if (piece.color == 1 and piece.row > 1) or (piece.color == 0 and piece.row > 5):
                        if self.pos_to_piece[(piece.row - 1, piece.col + 1)] is None:
                            pos = (piece.row - 2, piece.col + 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
                                destinies_.append(pos)
                    # right lower
                    if (piece.color == 1 and piece.row < 3) or (piece.color == 0 and piece.row < 8):
                        if self.pos_to_piece[(piece.row + 1, piece.col + 1)] is None:
                            pos = (piece.row + 2, piece.col + 2)
                            if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
                                destinies_.append(pos)

            elif piece.category == 'shi':
                # upper left
                if (piece.col == 3 and piece.row == 0) or (piece.col == 3 and piece.row == 7):
                    pos = (piece.row + 1, piece.col + 1)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
                        destinies_.append(pos)
                # upper right
                if (piece.col == 5 and piece.row == 0) or (piece.col == 5 and piece.row == 7):
                    pos = (piece.row + 1, piece.col - 1)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
                        destinies_.append(pos)
                # lower left
                if (piece.col == 3 and piece.row == 2) or (piece.col == 3 and piece.row == 9):
                    pos = (piece.row - 1, piece.col + 1)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
                        destinies_.append(pos)
                # lower right
                if (piece.col == 5 and piece.row == 2) or (piece.col == 5 and piece.row == 9):
                    pos = (piece.row - 1, piece.col - 1)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
                        destinies_.append(pos)
                # middle
                if (piece.col == 4 and piece.row == 1) or (piece.col == 4 and piece.row == 8):
                    pos_ = [(piece.row - 1, piece.col - 1), (piece.row - 1, piece.col + 1),
                            (piece.row + 1, piece.col - 1), (piece.row + 1, piece.col + 1)]
                    for pos in pos_:
                        if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
                            destinies_.append(pos)

            elif piece.category == 'jiang':
                # left
                if piece.col > 3:
                    pos = (piece.row, piece.col - 1)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
                        destinies_.append(pos)
                # right
                if piece.col < 5:
                    pos = (piece.row, piece.col + 1)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
                        destinies_.append(pos)
                # upper
                if piece.row > 0 or piece.row > 7:
                    pos = (piece.row - 1, piece.col)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
                        destinies_.append(pos)
                # lower
                if piece.row < 0 or piece.row < 9:
                    pos = (piece.row + 1, piece.col)
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
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
                                if piece__.color == 1 - piece.color:
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
                                if piece__.color == 1 - piece.color:
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
                                if piece__.color == 1 - piece.color:
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
                                if piece__.color == 1 - piece.color:
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
                    if self.pos_to_piece[pos] is None or self.pos_to_piece[pos].color == 1 - piece.color:
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
            self.pieces.pop(dst_index)
        self.pos_to_piece[(src_row, src_col)] = None
        self.pos_to_piece[(dst_row, dst_col)] = src_piece
        self.board_matrix[dst_row, dst_col] = src_piece.color * 10 + src_piece.get_cid()
        self.board_matrix[src_row, src_col] = 0
        src_piece.row = dst_row
        src_piece.col = dst_col
        self.next_turn = 1 - self.next_turn

    def get_result(self):
        prev_turn = 1 - self.next_turn
        if prev_turn == 0 and self.jiangblack not in self.pieces:
            return 0
        if prev_turn == 1 and self.jiangred not in self.pieces:
            return 1
        _, moves = self.feasible_moves()
        num_move = 0
        for moves_p in moves:
            num_move += len(moves_p)
        if num_move == 0:
            return prev_turn
        return 3

if __name__ == '__main__':
    board = Board()
    print(len(board.pieces))
    board.move(7, 1, 0, 1)
    print(len(board.pieces))
    pieces, moves = board.feasible_moves()
    print(moves)

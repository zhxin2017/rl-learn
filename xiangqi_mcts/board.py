import numpy as np

from piece import Piece

NROW = 10
NCOL = 9


class Board:
    def __init__(self, next_turn='red', state_str=None, my_color='red', maxstep=200):
        self.maxstep = maxstep
        self.pieces = []
        self.my_color = my_color
        self.next_turn = next_turn
        if state_str is not None:
            self.load_state(state_str)
        else:
            self.init_board()
        self.feasible_srcs, self.feasible_dsts = self.get_feasible_moves()
        self.src_row=None
        self.src_col=None
        self.dst_row=None
        self.dst_col=None
    
    def load_state(self, coords, next_turn, isdeads):
        # todo: pieces are not matched
        self.pieces = []
        board = [[Piece('none', 'none') for j in range(NCOL)] for i in range(NROW)]
        self.next_turn = next_turn
        categories = ['ju', 'ju', 'ma', 'ma', 'xiang', 'xiang', 'shi', 'shi', 'king', 
                'pao', 'pao', 'zu', 'zu', 'zu', 'zu', 'zu'] * 2
        for i, ((row, col), cate, isdead) in enumerate(zip(coords, categories, isdeads)):
            if i < 16:
                color = 'red'
                pid = i
            else:
                color = 'black'
                pid = i - 16

            piece = Piece(color, cate, row, col, pid, isdead)
            self.pieces.append(piece)
            if isdead:
                continue
            board[row][col] = piece
        self.board = board
        self.my_color = None
        if self.pieces[8].row > 2:
            self.my_color = 'red'
        else:
            self.my_color = 'black'
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

        red_ju1 = Piece('red', 'ju', row=row_ju_red, col=0, id=0)
        board[row_ju_red][0] = red_ju1
        red_ju2 = Piece('red', 'ju', row=row_ju_red, col=8, id=1)
        board[row_ju_red][8] = red_ju2
        red_ma1 = Piece('red', 'ma', row=row_ju_red, col=1, id=2)
        board[row_ju_red][1] = red_ma1
        red_ma2 = Piece('red', 'ma', row=row_ju_red, col=7, id=3)
        board[row_ju_red][7] = red_ma2
        red_xiang1 = Piece('red', 'xiang', row=row_ju_red, col=2, id=4)
        board[row_ju_red][2] = red_xiang1
        red_xiang2 = Piece('red', 'xiang', row=row_ju_red, col=6, id=5)
        board[row_ju_red][6] = red_xiang2
        red_shi1 = Piece('red', 'shi', row=row_ju_red, col=3, id=6)
        board[row_ju_red][3] = red_shi1
        red_shi2 = Piece('red', 'shi', row=row_ju_red, col=5, id=7)
        board[row_ju_red][5] = red_shi2
        red_king = Piece('red', 'king', row=row_ju_red, col=8)
        board[row_ju_red][4] = red_king
        red_pao1 = Piece('red', 'pao', row=row_pao_red, col=1, id=9)
        board[row_pao_red][1] = red_pao1
        red_pao2 = Piece('red', 'pao', row=row_pao_red, col=7, id=10)
        board[row_pao_red][7] = red_pao2
        red_zu1 = Piece('red', 'zu', row=row_zu_red, col=0, id=11)
        board[row_zu_red][0] = red_zu1
        red_zu2 = Piece('red', 'zu', row=row_zu_red, col=2, id=12)
        board[row_zu_red][2] = red_zu2
        red_zu3 = Piece('red', 'zu', row=row_zu_red, col=4, id=13)
        board[row_zu_red][4] = red_zu3
        red_zu4 = Piece('red', 'zu', row=row_zu_red, col=6, id=14)
        board[row_zu_red][6] = red_zu4
        red_zu5 = Piece('red', 'zu', row=row_zu_red, col=8, id=15)
        board[row_zu_red][8] = red_zu5

        black_ju1 = Piece('black', 'ju', row=row_ju_black, col=0, id=0)
        board[row_ju_black][0] = black_ju1
        black_ju2 = Piece('black', 'ju', row=row_ju_black, col=8, id=1)
        board[row_ju_black][8] = black_ju2
        black_ma1 = Piece('black', 'ma', row=row_ju_black, col=1, id=2)
        board[row_ju_black][1] = black_ma1
        black_ma2 = Piece('black', 'ma', row=row_ju_black, col=7, id=3)
        board[row_ju_black][7] = black_ma2
        black_xiang1 = Piece('black', 'xiang', row=row_ju_black, col=2, id=4)
        board[row_ju_black][2] = black_xiang1
        black_xiang2 = Piece('black', 'xiang', row=row_ju_black, col=6, id=5)
        board[row_ju_black][6] = black_xiang2
        black_shi1 = Piece('black', 'shi', row=row_ju_black, col=3, id=6)
        board[row_ju_black][3] = black_shi1
        black_shi2 = Piece('black', 'shi', row=row_ju_black, col=5, id=7)
        board[row_ju_black][5] = black_shi2
        black_king = Piece('black', 'king', row=row_ju_black, col=4, id=8)
        board[row_ju_black][4] = black_king
        black_pao1 = Piece('black', 'pao', row=row_pao_black, col=1, id=9)
        board[row_pao_black][1] = black_pao1
        black_pao2 = Piece('black', 'pao', row=row_pao_black, col=7, id=10)
        board[row_pao_black][7] = black_pao2
        black_zu1 = Piece('black', 'zu', row=row_zu_black, col=0, id=11)
        board[row_zu_black][0] = black_zu1
        black_zu2 = Piece('black', 'zu', row=row_zu_black, col=2, id=12)
        board[row_zu_black][2] = black_zu2
        black_zu3 = Piece('black', 'zu', row=row_zu_black, col=4, id=13)
        board[row_zu_black][4] = black_zu3
        black_zu4 = Piece('black', 'zu', row=row_zu_black, col=6, id=14)
        board[row_zu_black][6] = black_zu4
        black_zu5 = Piece('black', 'zu', row=row_zu_black, col=8, id=15)
        board[row_zu_black][8] = black_zu5

        self.pieces.append(red_ju1)
        self.pieces.append(red_ju2)
        self.pieces.append(red_ma1)
        self.pieces.append(red_ma2)
        self.pieces.append(red_xiang1)
        self.pieces.append(red_xiang2)
        self.pieces.append(red_shi1)
        self.pieces.append(red_shi2)
        self.pieces.append(red_king)
        self.pieces.append(red_pao1)
        self.pieces.append(red_pao2)
        self.pieces.append(red_zu1)
        self.pieces.append(red_zu2)
        self.pieces.append(red_zu3)
        self.pieces.append(red_zu4)
        self.pieces.append(red_zu5)

        self.pieces.append(black_ju1)
        self.pieces.append(black_ju2)
        self.pieces.append(black_ma1)
        self.pieces.append(black_ma2)
        self.pieces.append(black_xiang1)
        self.pieces.append(black_xiang2)
        self.pieces.append(black_shi1)
        self.pieces.append(black_shi2)
        self.pieces.append(black_king)
        self.pieces.append(black_pao1)
        self.pieces.append(black_pao2)
        self.pieces.append(black_zu1)
        self.pieces.append(black_zu2)
        self.pieces.append(black_zu3)
        self.pieces.append(black_zu4)
        self.pieces.append(black_zu5)

        for i in range(NROW):
            for j in range(NCOL):
                if board[i][j] is None:
                    board[i][j] = Piece('none', 'none')
        self.board = board
    
        self.last_move = None
        self.step = 0


    def shift_turn(self):
        if self.next_turn == 'red':
            self.next_turn = 'black'
        else:
            self.next_turn = 'red'


    def to_str(self):
        show = ''
        context = '\x1b[6;30;42m'
        context_end = '\x1b[0m'
        CRED = '\033[91m'
        CEND = '\033[0m'
        for i in range(NROW):
            for j in range(NCOL):
                piece = self.board[i][j]
                if piece.category == 'none':
                    if self.src_row is not None and i == self.src_row and j == self.src_col:
                        show += f'{context}　{context_end}'
                    else:
                        # show += '　'
                        show += '\033[33m十\033[0m'
                else:
                    if piece.color == 'red':
                        char = f'{CRED}{piece.get_char()}{CEND}'
                    else:
                        char = piece.get_char()
                    if self.dst_row is not None and i == self.dst_row and j == self.dst_col:
                        show += f'{context}{char}{context_end}'
                    else:
                        show += char

            show += '\n'
        return show
    
    def show_board(self):
        self.board_str = self.to_str()
        print(self.board_str)

    def check_king_facing(self, src_pos, dst_pos):
        dst_row, dst_col = dst_pos
        src_row, src_col = src_pos
        
        removed = self.move(src_row, src_col, dst_row, dst_col, update_feasible=False)
        if removed.category == 'king':
            self.restore(removed, src_row, src_col, dst_row, dst_col, update_feasible=False)
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
            self.restore(removed, src_row, src_col, dst_row, dst_col, update_feasible=False)
            return False

        for i in range(king1_row + 1, king2_row):
            if self.board[i][king1_col].category != 'none':
                self.restore(removed, src_row, src_col, dst_row, dst_col, update_feasible=False)
                return False

        self.restore(removed, src_row, src_col, dst_row, dst_col, update_feasible=False)
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
                            if r < 9:
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

    def move(self, src_row, src_col, dst_row, dst_col, update_feasible=True):
        src_piece = self.board[src_row][src_col]
        dst_piece = self.board[dst_row][dst_col]
        self.board[dst_row][dst_col] = src_piece
        src_piece.row = dst_row
        src_piece.col = dst_col
        self.board[src_row][src_col] = Piece('none', 'none')
        removed = dst_piece
        if removed.category != 'none':
            removed.isdead = 1
        self.shift_turn()
        if update_feasible:
            self.feasible_srcs, self.feasible_dsts = self.get_feasible_moves()
            self.src_row = src_row
            self.src_col = src_col
            self.dst_row = dst_row
            self.dst_col = dst_col
        self.step += 1
        return removed 

    def restore(self, removed_piece, src_row, src_col, dst_row, dst_col, update_feasible=True):
        moved_piece = self.board[dst_row][dst_col]
        self.board[src_row][src_col] = moved_piece
        moved_piece.row = src_row
        moved_piece.col = src_col
        self.board[dst_row][dst_col] = removed_piece
        if removed_piece.category != 'none':
            removed_piece.isdead = 0
        self.shift_turn()
        self.step -= 1
        if update_feasible:
            self.feasible_srcs, self.feasible_dsts = self.get_feasible_moves()
            

    def get_result(self):
        if self.pieces[8].isdead:
            return 'black'
        if self.pieces[24].isdead:
            return 'red'

        num_move = 0
        for moves_p in self.feasible_dsts:
            num_move += len(moves_p)
        if num_move == 0:
            return 'red' if self.next_turn == 'black' else 'black'
        if self.step >= self.maxstep:
            return 'draw'
        return 'going'


if __name__ == '__main__':
    import random
    board = Board(my_color='black', next_turn='black')
    board.move(9, 2, 7, 4)
    board.show_board()
    board.move(2, 1, 2, 4)
    board.show_board()
    board.move(9, 1, 7, 2)
    board.show_board()
    board.move(2, 4, 6, 4)
    board.show_board()
    board.move(9, 0, 9, 1)
    board.show_board()

    import copy
    srcs, dsts = board.get_feasible_moves()
    for i in range(len(srcs)):
        src_row, src_col = srcs[i]
        for dst_row, dst_col in dsts[i]:
            board_ = copy.deepcopy(board)
            board_.move(src_row, src_col, dst_row, dst_col)
            print(f'Move from ({src_row}, {src_col}) to ({dst_row}, {dst_col}), result: {board_.get_result()}')
            board_.show_board()

    

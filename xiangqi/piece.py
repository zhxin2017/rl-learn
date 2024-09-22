from config import piece_category_to_cid, color_str_to_id

char_dict = {
    'red': {
        'ju': '俥',
        'ma': '傌',
        'xiang': '相',
        'shi': '仕',
        'king': '帥',
        'pao': '炮',
        'zu': '兵'
    },
    'black': {
        'ju': '車',
        'ma': '馬',
        'xiang': '象',
        'shi': '士',
        'king': '将',
        'pao': '砲',
        'zu': '卒'
    }
}


class Piece:
    def __init__(self, row, col, color, category):
        self.row = row
        self.col = col
        self.color = color
        self.category = category

    def get_cid(self):
        return piece_category_to_cid[self.category]

    def get_color_id(self):
        return color_str_to_id[self.color]

    def get_char(self):
        return char_dict[self.color][self.category]

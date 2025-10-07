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
    def __init__(self, color, category=None):
        self.color = color
        self.category = category

    def get_cid(self):
        cid = piece_category_to_cid[self.category]
        if self.color == 'black' and cid != 0:
            cid = cid + 7
        return cid

    def get_char(self):
        return char_dict[self.color][self.category]

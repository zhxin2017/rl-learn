import torch
from torch import nn
import tsfm
import torch.nn.functional as F


class Evaluator(nn.Module):
    def __init__(self, n_layer, dmodel=256):
        super().__init__()
        self.category_emb_m = nn.Embedding(8, dmodel)
        self.color_emb_m = nn.Embedding(2, dmodel)
        self.pos_emb_m = nn.Embedding(90, dmodel)
        self.turn_emb_m = nn.Embedding(2, dmodel)
        self.result_query_emb_m = nn.Embedding(1, dmodel)
        self.encoder_layers = nn.ModuleList()
        for i in range(n_layer):
            encoder_layer = tsfm.AttnLayer(dmodel, dmodel, dmodel, n_head=dmodel // 64)
            self.encoder_layers.append(encoder_layer)
        self.result_reg = nn.Linear(dmodel, 3)

    def forward(self, cid, color, turn):
        b = cid.shape[0]
        category_emb = self.category_emb_m(cid)
        color_emb = self.color_emb_m(color)
        color_mask = cid > 0
        color_emb = color_emb * color_mask.view(b, 10, 9, 1)
        turn_emb = self.turn_emb_m(turn).view(b, 1, -1)
        x = category_emb + color_emb
        x = x.view(b, 90, -1)
        pos_emb = self.pos_emb_m(torch.arange(90, device=cid.device)).view(1, 90, -1)
        x = x + pos_emb
        result_query = self.result_query_emb_m(torch.zeros([b], dtype=torch.int, device=cid.device)).view(b, 1, -1)
        result_query = result_query + turn_emb
        x = torch.concat([x, result_query], dim=1)
        for enc in self.encoder_layers:
            x = enc(x, x, x, x)
        result = self.result_reg(x[:, -1])
        return result

if __name__ == '__main__':
    import dataset
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    stat_file = '/Users/zx/Documents/rl-exp/xiangqi/stat.0.json'
    ds = dataset.Ds(stat_file)
    dl = DataLoader(ds, batch_size=2)
    model_ = Evaluator(12, 256)
    for category, color, next_turn, probs in tqdm(dl):
        probs_pred = model_(category, color, next_turn)
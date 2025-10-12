import torch
from torch import nn
import tsfm
import torch.nn.functional as F


class Evaluator(nn.Module):
    def __init__(self, n_layer, dmodel=256, dhead=64):
        super().__init__()
        self.dmodel = dmodel
        self.category_emb_m = nn.Embedding(15, dmodel)
        self.color_emb_m = nn.Embedding(3, dmodel)
        self.row_emb_m = nn.Embedding(10, dmodel)
        self.col_emb_m = nn.Embedding(9, dmodel)
        self.turn_emb_m = nn.Embedding(2, dmodel)
        self.result_query_emb_m = nn.Embedding(1, dmodel)
        self.encoder_layers = nn.ModuleList()
        for i in range(n_layer):
            encoder_layer = tsfm.Block(dmodel, dhead)
            self.encoder_layers.append(encoder_layer)
        self.result_reg = nn.Linear(dmodel, 1)

    def forward(self, cids, next_turn):
        b = cids.shape[0]
        category_emb = self.category_emb_m(cids)
        row_emb = self.row_emb_m(torch.arange(10, device=cids.device)).view(1, 10, 1, self.dmodel)
        col_emb = self.col_emb_m(torch.arange(9, device=cids.device)).view(1, 1, 9, self.dmodel)
        x = category_emb + row_emb + col_emb
        x = x.view(b, 90, -1)
        result_query = (self.result_query_emb_m(torch.tensor([0], dtype=torch.int, device=cids.device)))
        result_query = result_query.view(1, 1, self.dmodel).repeat(b, 1, 1)
        next_turn_emb = self.turn_emb_m(next_turn).view(b, 1, self.dmodel)
        x = torch.concat([x, next_turn_emb, result_query], dim=1)
        for enc in self.encoder_layers:
            x = enc(x, x, x)
        result = self.result_reg(x[:, -1])
        result = (torch.sigmoid(result) - 0.5) * 2
        return result


if __name__ == '__main__':
    import dataset
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    stat_file = '/Users/zx/Documents/rl-exp/xiangqi/stat.0.json'
    ds = dataset.Ds(stat_file)
    dl = DataLoader(ds, batch_size=2)
    model_ = Evaluator(24, 512, 64)
    for category, color, next_turn, probs in tqdm(dl):
        probs_pred = model_(category, color, next_turn)

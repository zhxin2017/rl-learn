import torch
from torch import nn
import tsfm
import torch.nn.functional as F


class Evaluator(nn.Module):
    def __init__(self, n_layer, dmodel=256, dhead=64):
        super().__init__()
        self.dmodel = dmodel
        self.category_emb_m = nn.Embedding(7, dmodel)
        self.color_emb_m = nn.Embedding(2, dmodel)
        self.row_emb_m = nn.Embedding(10, dmodel)
        self.col_emb_m = nn.Embedding(9, dmodel)
        self.turn_emb_m = nn.Embedding(2, dmodel)
        self.result_query_emb_m = nn.Embedding(1, dmodel)
        self.isdead_emb_m = nn.Embedding(1, dmodel)
        self.encoder_layers = nn.ModuleList()
        for i in range(n_layer):
            encoder_layer = tsfm.Block(dmodel, dhead)
            self.encoder_layers.append(encoder_layer)
        self.result_reg = nn.Linear(dmodel, 1)

        self.act_linear = nn.ModuleList()

        self.act_linear_ju = nn.Linear(dmodel, 34)
        self.act_linear_ma = nn.Linear(dmodel, 8)
        self.act_linear_xiang = nn.Linear(dmodel, 4)
        self.act_linear_shi = nn.Linear(dmodel, 4)
        self.act_linear_king = nn.Linear(dmodel, 4)
        self.act_linear_pao = nn.Linear(dmodel, 34)
        self.act_linear_zu = nn.Linear(dmodel, 4)
        self.act_linear.append(self.act_linear_ju)
        self.act_linear.append(self.act_linear_ju)
        self.act_linear.append(self.act_linear_ma)
        self.act_linear.append(self.act_linear_ma)
        self.act_linear.append(self.act_linear_xiang)
        self.act_linear.append(self.act_linear_xiang)
        self.act_linear.append(self.act_linear_shi)
        self.act_linear.append(self.act_linear_shi)
        self.act_linear.append(self.act_linear_king)
        self.act_linear.append(self.act_linear_pao)
        self.act_linear.append(self.act_linear_pao)
        self.act_linear.append(self.act_linear_zu)
        self.act_linear.append(self.act_linear_zu)
        self.act_linear.append(self.act_linear_zu)
        self.act_linear.append(self.act_linear_zu)
        self.act_linear.append(self.act_linear_zu)


    def forward(self, coords, next_turns, isdeads):
        b = coords.shape[0]
        cids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 6, 6, 6, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 6, 6, 6], dtype=torch.int, device=coords.device).view(1, 32).repeat(b, 1)
        colors = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int, device=coords.device).view(1, 32).repeat(b, 1)
        rows_emb = self.row_emb_m(coords[..., 0])
        cols_emb = self.col_emb_m(coords[..., 1])
        isdeads_emb = self.isdead_emb_m(torch.zeros(b, 32, dtype=torch.int, device=coords.device))
        category_emb = self.category_emb_m(cids)
        colors_emb = self.color_emb_m(colors)
        isdeads = isdeads.unsqueeze(-1)
        x = category_emb + colors_emb + (rows_emb + cols_emb) * (1 - isdeads) + isdeads_emb * isdeads
        result_query = self.result_query_emb_m(torch.zeros(b, 1, dtype=torch.int, device=coords.device))
        # next_turn_emb = self.turn_emb_m(next_turns).view(b, 1, self.dmodel)
        next_turns = next_turns.unsqueeze(-1)
        next_turn_emb = self.turn_emb_m(next_turns)
        x = torch.concat([x, next_turn_emb, result_query], dim=1)
        for enc in self.encoder_layers:
            x = enc(x, x, x)
        result = self.result_reg(x[:, -1])
        result = (torch.sigmoid(result) - 0.5) * 2
        action_logits = []
        x = x[:, :-2].view(b, 2, 16, self.dmodel)[torch.arange(b), next_turns.view(b)]
        for j in range(len(self.act_linear)):
            act_logit = self.act_linear[j](x[:, j])
            action_logits.append(act_logit)
        act_logits = torch.cat(action_logits, dim=-1)
        return result, act_logits


if __name__ == '__main__':
    import dataset
    from torch.utils.data import DataLoader

    stat_file = '/Users/zx/Documents/rl-exp/xiangqi/stat.0.json'
    ds = dataset.Ds(stat_file)
    dl = DataLoader(ds, batch_size=2)
    model_ = Evaluator(24, 512, 64)
    for category, color, next_turn, probs in dl:
        probs_pred = model_(category, color, next_turn)

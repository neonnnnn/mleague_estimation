import torch
from torch.nn import Embedding
from torch.nn.parameter import Parameter
from torch.nn import Module
from sinkhorn_rank_sort import SinkhornRankSort


class BT(Module):
    def __init__(self, n_players=30):
        super().__init__()
        self.embed = Embedding(n_players, 1)

    def forward(self, X):
        transform = 4.0 * torch.eye(4) -1 * torch.ones((4, 4))
        strength = torch.squeeze(self.embed.forward(X), dim=-1)
        return torch.mm(strength, transform.to(X.device))
    

class SinkhornBT(Module):
    def __init__(self, n_players=30, eps=1e-3, max_iter=1000,
                 tau=0.1, tol=1e-2, p=2):
        super().__init__()
        self.bt = BT(n_players)

        self.sinkhorn = SinkhornRankSort(
            eps=eps, max_iter=max_iter, tau=tau,
            tol=tol, p=p, sort=False, rank=False)
        
    def forward(self, X):
        raw_point = self.bt(X)
        ranking_point = torch.tensor([-35, -15, 5, 45]).to(X.device)
        prob_ranking = self.sinkhorn(raw_point) * 4.0
        expected_ranking_point = torch.sum(
            prob_ranking * ranking_point, 
            dim=2, keepdims=False
        ) 
        return raw_point + expected_ranking_point
import torch
from torch.nn import Embedding
from torch.nn.parameter import Parameter
from torch.nn import Module
from math import sqrt, log


def softmin(M, eps, dim=-1, keepdim=False):
    result = - eps * torch.logsumexp(-M/eps, dim=dim, keepdim=keepdim)
    return result


class SinkhornRankSort(Module):
    def __init__(self, eps=1e-2, max_iter=1000, 
                 tau=1.0, tol=1e-5, p=2,rank=True, sort=True):
        super().__init__()
        self.max_iter = max_iter
        self.eps = eps
        self.tol = tol
        self.p = p
        self.rank = rank
        self.sort = sort
        self.tau = tau
    
    def _standarize(self, X):
        X_tilde = X - torch.sum(X, dim=1, keepdim=True)
        X_tilde = torch.nn.functional.normalize(X_tilde, p=2)
        X_tilde *= sqrt(X.shape[1])
        X_tilde = torch.sigmoid(X_tilde / self.tau)
        return X_tilde
        
    def _stopping_criterion(self, P):
        diff_a = torch.sum(P, dim=1) - (1.0/P.shape[1])
        diff_b = torch.sum(P, dim=2) - (1.0/P.shape[1])
        max_delta_a = torch.max(torch.abs(diff_a))
        max_delta_b = torch.max(torch.abs(diff_b))
        return max(max_delta_a.item(), max_delta_b.item()) < self.tol

    def forward_naive(self, X):
        n_input = X.shape[1]
        y = torch.arange(n_input, dtype=torch.float32) / (n_input-1)
        b_bar = (1+torch.arange(n_input, dtype=torch.float32))/n_input
        alpha = torch.zeros(X.shape)
        beta = torch.zeros(X.shape)
        
        X_tilde = self._standarize(X)
        C = torch.abs(X_tilde[:, :, None] - y[None, None, :]) ** self.p

        K = torch.exp(- C / self.eps)
        U = torch.ones(X.shape)
        V = torch.zeros(X.shape)
        for it in range(self.max_iter):
            V = 1.0 / torch.sum(K*U[:, :, None], dim=1)
            V /= n_input
            U = 1.0 / torch.sum(K*V[:, None,:], dim=2)
            U /= n_input
            
            P = U[:, :, None] * K * V[:, None, :]
            is_converged = self._stopping_criterion(P)
            if is_converged:
                break
                
        if not (self.rank and self.sort):
            return P
        else:
            output = [P]
            if self.rank:
                R = n_input * n_input * U * torch.sum(K * (V*b_bar)[:, None, :], dim=-1, keepdim=False)
                output.append(R)
            if self.sort:
                S = n_input * V * torch.sum(K.transpose(1, 2) * (U*X)[:, None, :], dim=-1, keepdim=False)
                output.append(S)

            return output
        
    def forward(self, X):
        device = X.device
        n_input = X.shape[1]
        y = torch.arange(n_input, dtype=torch.float32).to(device) / (n_input-1)
        alpha = torch.zeros(X.shape).to(device)
        beta = torch.zeros(X.shape).to(device)
        b_bar = (1+torch.arange(n_input, dtype=torch.float32).to(device))/n_input

        X_tilde = self._standarize(X)
        C = torch.abs(X_tilde[:, :, None] - y[None, None, :]) ** self.p
        
        for it in range(self.max_iter):
            M = C - alpha[:, :, None] - beta[:, None, :]
            beta += -self.eps * log(n_input) + softmin(M, self.eps, dim=1)            
            
            M = C - alpha[:, :, None] - beta[:, None, :]
            alpha += -self.eps * log(n_input) + softmin(M, self.eps, dim=2)
            
            #log_P = log(n_input) - M / self.eps
            #P = torch.exp(log_P)
            M = C - alpha[:, :, None] - beta[:, None, :]
            P = torch.exp(- M / self.eps)
            is_converged = self._stopping_criterion(P)
            if is_converged:
                break
                
        if not (self.rank and self.sort):
            return P
        else:
            output = [P]
            if self.rank:
                R = n_input * n_input * torch.sum(P*b_bar, dim=-1, keepdim=False)
                output.append(R)
            if self.sort:
                S = n_input * torch.sum(P.transpose(1, 2) * X[:, None, :], dim=-1, keepdim=False)
                output.append(S)

            return output
        
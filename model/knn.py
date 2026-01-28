import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_kNN(nn.Module):
    def __init__(self,args,
        nlayers: int,
        isize: int,
        hsize: int,
        osize: int,
        k: int,
        non_linearity: str = "relu",
        i: float = 1.0,
        mlp_act: str = "relu",
    ):
        super().__init__()

        layers = []
        if nlayers == 1:
            layers.append(nn.Linear(isize, hsize))
        else:
            layers.append(nn.Linear(isize, hsize))
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hsize, hsize))
            layers.append(nn.Linear(hsize, osize))
        self.layers = nn.ModuleList(layers)

        self.input_dim = isize
        self.output_dim = osize
        self.k = k
        self.non_linearity = non_linearity
        self.slope = i
        self.mlp_act = mlp_act

        if self.input_dim == self.output_dim:
            for layer in self.layers:
                if isinstance(layer, nn.Linear) and layer.weight.shape[0] == layer.weight.shape[1] == self.input_dim:
                    with torch.no_grad():
                        layer.weight.copy_(torch.eye(self.input_dim))
                        if layer.bias is not None:
                            layer.bias.zero_()

    def internal_forward(self, h: torch.Tensor) -> torch.Tensor:
        for li, layer in enumerate(self.layers):
            residual = h
            h = layer(h)

            if li != (len(self.layers) - 1):
                h = h + residual
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = torch.tanh(h)
        return h

    def forward(self, timeseries: torch.Tensor, pearson: torch.Tensor) -> torch.Tensor:
        B, N, T = timeseries.shape
        eps = 1e-6
        # z = F.normalize(timeseries, p=2, dim=-1)  # [N, d]
        mean = timeseries.mean(dim=-1, keepdim=True)
        std = timeseries.std(dim=-1, keepdim=True)
        z = (timeseries - mean) / (std + 1e-8)
        
        embeddings = self.internal_forward(z)  # [N, d]
        cosine = embeddings @ embeddings.transpose(-1, -2)
        # cosine = F.normalize(cosine, p=2, dim=-1)
        # cosine = (cosine + cosine.transpose(-1, -2)) / 2

        A = top_k_dense(cosine, self.k)  # [N, N]
        A = F.normalize(A, p=2, dim=-1)
        A = apply_non_linearity(A, self.non_linearity, self.slope)
        return A

# -------------------- helpers --------------------
def apply_non_linearity(x: torch.Tensor, non_linearity: str = "relu", slope: float = 1.0) -> torch.Tensor:
    if non_linearity == "relu":
        return F.relu(x)
    elif non_linearity == "elu":
        return F.elu(x) + 1.0
    elif non_linearity == "exp":
        return torch.exp(slope * x)
    else:
        return x

def top_k_dense(S: torch.Tensor, k: int, include_self: bool = False, fill_val: float = 1e-6) -> torch.Tensor:
    N = S.size(-1)
    kk = k + 1 if include_self else k

    if not include_self:
        if S.dim() == 2:
            mask = torch.eye(N, device=S.device, dtype=torch.bool)
        else:  # [B, N, N]
            mask = torch.eye(N, device=S.device, dtype=torch.bool).unsqueeze(0)
        S = S.masked_fill(mask, 1e-6)

    vals, idx = torch.topk(S, k=kk, dim=-1)

    out = torch.full_like(S, fill_val)
    out.scatter_(dim=-1, index=idx, src=vals)
    return out



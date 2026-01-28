import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphDAE(nn.Module):
    def __init__(self,args, in_dim, hidden_dim=64, num_layers=2,
                 activation='relu', dropout=0.0):
        super().__init__()
        assert num_layers >= 1

        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [in_dim]
        acts = [activation] * (num_layers - 1) + ['none']
        for din, dout, act in zip(dims[:-1], dims[1:], acts):
            layers.append(DenseGCNLayer(din, dout, activation=act, dropout=dropout))
        self.layers = nn.ModuleList(layers)
        self.noise_percent = args.noise_percent

    def forward(self, A: torch.Tensor, node_features: torch.Tensor):
        # A: [B,N,N]; noisy_x: [B,N,T]
        A = normalize_adj_dense(A)
        if self.training:
            noise = make_mask(node_features,self.noise_percent)
            noise_features = apply_noise(node_features, noise)
            h = noise_features
        else:
            noise = None
            h = node_features
        for gcn in self.layers:
            h = gcn(A, h)

        return h, noise  # [B,N,T]


def dae_recon_loss(x_clean,x_recon, noise, loss_type='mse') -> torch.Tensor:
    eps = 1e-8
    if loss_type == 'mse':
        diff = (x_recon - x_clean) ** 2
        loss = (diff * noise).sum() / (noise.sum() + eps)
    elif loss_type == 'bce':
        pred = torch.sigmoid(x_recon)
        bce = -(x_clean * torch.log(pred + eps) + (1 - x_clean) * torch.log(1 - pred + eps))
        loss = (bce * noise).sum() / (noise.sum() + eps)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    return loss




class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation='relu', dropout=0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=True)
        self.activation = activation
        self.dropout = dropout

    def forward(self, A: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        # [B,N,N] @ [B,N,F] -> [B,N,F]
        H = torch.matmul(A, H)
        H = self.lin(H)
        if self.activation == 'relu':
            H = F.relu(H)
        elif self.activation == 'tanh':
            H = torch.tanh(H)
        elif self.activation == 'leaky_relu':
            H = F.leaky_relu(H)
        elif self.activation in (None, 'none'):
            pass
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        if self.dropout > 0:
            H = F.dropout(H, p=self.dropout, training=self.training)
        return H


def normalize_adj_dense(A: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    B, N, _ = A.shape
    I = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)  # [B,N,N]
    A = A + I

    deg = A.sum(dim=-1)                           # [B, N]
    inv_sqrt = (deg + eps).pow(-0.5)              # [B, N]
    D_inv_sqrt = torch.diag_embed(inv_sqrt)       # [B, N, N]
    return D_inv_sqrt @ A @ D_inv_sqrt



def make_mask(x: torch.Tensor, r: float = 0.1) -> torch.Tensor:
    return (torch.rand_like(x) < r).to(x.dtype)

def apply_noise(x: torch.Tensor,
                mask: torch.Tensor,
                mode: str = "normal",
                sigma: float = 0.2) -> torch.Tensor:
    if mode == "mask":
        noisy = x * (1.0 - mask)
    elif mode == "normal":
        noise = torch.randn_like(x) * sigma
        noisy = x + noise * mask
    else:
        raise ValueError(f"Unknown noise mode: {mode}")
    return noisy




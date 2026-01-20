import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Helper: Sparsemax ---
def sparsemax(v, dim=-1):
    number_of_logits = v.shape[dim]
    v_sorted, _ = torch.sort(v, dim=dim, descending=True)
    cumsum_v = torch.cumsum(v_sorted, dim=dim)
    range_values = torch.arange(1, number_of_logits + 1, device=v.device).float()

    bound = 1 + range_values * v_sorted
    condition = bound > cumsum_v
    k_x = condition.long().sum(dim=dim, keepdim=True)

    tau = (torch.gather(cumsum_v, dim=dim, index=k_x - 1) - 1) / k_x.float()
    return torch.clamp(v - tau, min=0)

# --- Model A: SpaDE (Paper version) ---
class SpaDE(nn.Module):
    def __init__(self, input_dim, latent_dim, initial_scale=None, fix_scale=False):
        super().__init__()
        # Enkoder to prototypy
        self.encoder_prototypes = nn.Parameter(torch.randn(latent_dim, input_dim) * 0.02)

        # --- ZMIANA ZGODNA Z PAPEREM ---
        # 1. Domyślna inicjalizacja: 1 / (2 * input_dim)
        if initial_scale is None:
            target_scale = 1.0 / (2.0 * input_dim)
        else:
            target_scale = initial_scale

        # 2. Parametryzacja Softplus.
        # Musimy zainicjować "surowy" parametr tak, aby po softplus dał target_scale.
        # Softplus(x) = log(1 + exp(x))  =>  x = log(exp(target) - 1)
        # Dodajemy epsilon dla stabilności numerycznej
        init_raw_param = np.log(np.exp(target_scale) - 1.0 + 1e-6)

        raw_scale_tensor = torch.tensor(init_raw_param, dtype=torch.float32)

        if fix_scale:
            self.register_buffer("scale_param", raw_scale_tensor)
        else:
            self.scale_param = nn.Parameter(raw_scale_tensor)

        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)

    def get_dist(self, x):
        x_norm = (x**2).sum(1).view(-1, 1)
        w_norm = (self.encoder_prototypes**2).sum(1).view(1, -1)
        dist = x_norm + w_norm - 2.0 * torch.matmul(x, self.encoder_prototypes.t())
        return dist

    def forward(self, x):
        dists = self.get_dist(x)

        # --- ZMIANA: Softplus zamiast Exp ---
        scale = F.softplus(self.scale_param)

        # Sparsemax na ujemnym dystansie
        z = sparsemax(-scale * dists, dim=-1)
        x_hat = self.decoder(z)
        return x_hat, z

# --- Model B: TopK SAE ---
class TopKSAE(nn.Module):
    def __init__(self, input_dim, latent_dim, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)

        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.t().clone()

    def forward(self, x):
        pre_acts = self.encoder(x)
        pre_acts = F.relu(pre_acts)

        topk_values, topk_indices = torch.topk(pre_acts, k=self.k, dim=-1)
        z = torch.zeros_like(pre_acts)
        z.scatter_(-1, topk_indices, topk_values)

        x_hat = self.decoder(z)
        return x_hat, z

# --- Model C: ReLU SAE ---
class ReLUSAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)

        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.t().clone()
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, p=2, dim=0)

    def forward(self, x):
        pre_acts = self.encoder(x)
        z = F.relu(pre_acts)
        x_hat = self.decoder(z)
        return x_hat, z

# --- Model D: SzpaDeLDiag (Diagonal Mahalanobis with shared projection for prototypes) ---
class SzpaDeLDiag(SpaDE):
    def __init__(self, input_dim, latent_dim, initial_scale=None, fix_scale=False):
        super().__init__(input_dim, latent_dim, initial_scale, fix_scale)
        self.raw_diag = nn.Parameter(torch.zeros(input_dim))  # [D]

    def get_dist(self, x):
        w = F.softplus(self.raw_diag)  # [D]
        xw = x * w.sqrt()  # [B, D]
        mw = self.encoder_prototypes * w.sqrt()  # [L, D]
        x_norm = (xw**2).sum(dim=1, keepdim=True)  # [B, 1]
        m_norm = (mw**2).sum(dim=1, keepdim=True).t()  # [1, L]
        return x_norm + m_norm - 2.0 * (xw @ mw.t())  # [B, L]


# --- Model E: SzpaDeLDiagLocal (Diagonal Mahalanobis with separate projection for prototypes) ---
class SzpaDeLDiagLocal(SpaDE):
    def __init__(self, input_dim, latent_dim, initial_scale=None, fix_scale=False):
        super().__init__(input_dim, latent_dim, initial_scale, fix_scale)
        self.raw_diag = nn.Parameter(torch.zeros(latent_dim, input_dim))  # [L, D]

    def get_dist(self, x):
        w = F.softplus(self.raw_diag)  # [L, D]
        x2 = (x**2)  # [B, D]
        term1 = x2 @ w.t()  # [B, L]
        term2 = (w * (self.encoder_prototypes**2)).sum(dim=1, keepdim=True).t()  # [1, L]
        cross = x @ (w * self.encoder_prototypes).t()  # [B, L]
        return term1 + term2 - 2.0 * cross  # [B, L]

# --- Model F: SzpaDeLFull (Full Mahalanobis with shared projection for prototypes) ---
class SzpaDeL(SpaDE):
    def __init__(self, input_dim, latent_dim, initial_scale=None, fix_scale=False):
        super().__init__(input_dim, latent_dim, initial_scale, fix_scale)
        self.metric_factor = nn.Parameter(torch.eye(input_dim))  # [D, D]

    def get_dist(self, x):
        Ax = x @ self.metric_factor.t()  # [B, D]
        Am = self.encoder_prototypes @ self.metric_factor.t()  # [L, D]
        x_norm = (Ax**2).sum(dim=1, keepdim=True)  # [B, 1]
        m_norm = (Am**2).sum(dim=1, keepdim=True).t()  # [1, L]
        return x_norm + m_norm - 2.0 * (Ax @ Am.t())  # [B, L]


# --- Model G: SzpaDeLFullLocal (Full Mahalanobis with separate projection for prototypes) - DOESN'T WORK FOR ME, TOO MUCH MEMORY NEEDED  ---
class SzpaDeLLocal(SpaDE):
    def __init__(self, input_dim, latent_dim, initial_scale=None, fix_scale=False):
        super().__init__(input_dim, latent_dim, initial_scale, fix_scale)
        self.metric_factor = nn.Parameter(
            torch.stack([torch.eye(input_dim) for _ in range(latent_dim)], dim=0)
        )  # [L, D, D]

    def get_dist(self, x):
        Ax = torch.einsum("bd,ldk->blk", x, self.metric_factor.transpose(-1, -2))  # [B, L, D]
        Am = torch.einsum("ld,ldk->lk", self.encoder_prototypes, self.metric_factor.transpose(-1, -2))  # [L, D]
        x_norm = (Ax**2).sum(dim=-1)  # [B, L]
        m_norm = (Am**2).sum(dim=-1).unsqueeze(0)  # [1, L]
        cross = torch.einsum("blk,lk->bl", Ax, Am)  # [B, L]
        return x_norm + m_norm - 2.0 * cross  # [B, L]

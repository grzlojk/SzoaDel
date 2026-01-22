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
        self.encoder_prototypes = nn.Parameter(
            torch.randn(latent_dim, input_dim) * 0.02
        )

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
    # mahalonobis distance with non-zero elements only on the Diagonal
    # identical distance calculation for each prototype
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
    # mahalonobis distance with non-zero elements only on the Diagonal
    # DIFFERENT distance calculation for each prototype
    def __init__(self, input_dim, latent_dim, initial_scale=None, fix_scale=False):
        super().__init__(input_dim, latent_dim, initial_scale, fix_scale)
        self.raw_diag = nn.Parameter(torch.zeros(latent_dim, input_dim))  # [L, D]

    def get_dist(self, x):
        w = F.softplus(self.raw_diag)  # [L, D]
        x2 = x**2  # [B, D]
        term1 = x2 @ w.t()  # [B, L]
        term2 = (
            (w * (self.encoder_prototypes**2)).sum(dim=1, keepdim=True).t()
        )  # [1, L]
        cross = x @ (w * self.encoder_prototypes).t()  # [B, L]
        return term1 + term2 - 2.0 * cross  # [B, L]


# --- Model F: SzpaDeLFull (Full Mahalanobis with shared projection for prototypes) ---
class SzpaDeL(SpaDE):
    # real mahalonobis distance, identical for each prototype
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
    # real mahalonobis distance, DIFFERENT for each prototype
    def __init__(self, input_dim, latent_dim, initial_scale=None, fix_scale=False):
        super().__init__(input_dim, latent_dim, initial_scale, fix_scale)
        self.metric_factor = nn.Parameter(
            torch.stack([torch.eye(input_dim) for _ in range(latent_dim)], dim=0)
        )  # [L, D, D]

    def get_dist(self, x):
        Ax = torch.einsum(
            "bd,ldk->blk", x, self.metric_factor.transpose(-1, -2)
        )  # [B, L, D]
        Am = torch.einsum(
            "ld,ldk->lk", self.encoder_prototypes, self.metric_factor.transpose(-1, -2)
        )  # [L, D]
        x_norm = (Ax**2).sum(dim=-1)  # [B, L]
        m_norm = (Am**2).sum(dim=-1).unsqueeze(0)  # [1, L]
        cross = torch.einsum("blk,lk->bl", Ax, Am)  # [B, L]
        return x_norm + m_norm - 2.0 * cross  # [B, L]


class SzpaDeLRank1(SpaDE):
    """
    Rank-1 Local Mahalanobis Distance.
    Mathematically: Dist(x, m) = ||x - m||^2 + ((x - m)^T u)^2

    Equivalent to learning a matrix M = I + u*u.T for each prototype.
    This allows the prototype to rotate its sensitivity axis without O(D^2) memory.
    """

    def __init__(self, input_dim, latent_dim, initial_scale=None, fix_scale=False):
        super().__init__(input_dim, latent_dim, initial_scale, fix_scale)
        # We learn one directional vector 'u' per prototype.
        # Initialization is small to start close to standard Euclidean distance.
        self.metric_vectors = nn.Parameter(torch.randn(latent_dim, input_dim) * 0.01)

    def get_dist(self, x):
        x_norm = (x**2).sum(dim=1, keepdim=True)  # [B, 1]
        m_norm = (self.encoder_prototypes**2).sum(dim=1).view(1, -1)  # [1, L]

        cross_euclidean = x @ self.encoder_prototypes.t()  # [B, L]

        euclidean_dist = x_norm + m_norm - 2.0 * cross_euclidean  # [B, L]

        # Project input batch onto the directional vectors
        # x: [B, D], u: [L, D] -> x @ u.T -> [B, L]
        # This tells us: "How strong is input x in prototype L's special direction?"
        xu = x @ self.metric_vectors.t()

        # Project prototypes onto their own directional vectors
        # Element-wise mul then sum: (m * u).sum -> [L]
        mu = (
            (self.encoder_prototypes * self.metric_vectors).sum(dim=1).view(1, -1)
        )  # [1, L]

        rank1_correction = (xu - mu) ** 2

        return euclidean_dist + rank1_correction


class SzpaDeLMultiHead(SpaDE):
    """
    Factorized / Multi-Head Mahalanobis Distance.
    Instead of L different matrices, we learn H shared matrices.
    Each prototype is a weighted combination of these H metrics.
    """

    def __init__(
        self, input_dim, latent_dim, initial_scale=None, fix_scale=False, num_heads=4
    ):
        super().__init__(input_dim, latent_dim, initial_scale, fix_scale)
        self.num_heads = num_heads

        self.metric_heads = nn.Parameter(
            torch.stack([torch.eye(input_dim) for _ in range(num_heads)], dim=0)
        )  # [H, D, D]

        # Learn weights: which prototype prefers which head?
        self.proto_head_logits = nn.Parameter(torch.randn(latent_dim, num_heads) * 0.1)

    def get_dist(self, x):
        B = x.shape[0]
        L = self.encoder_prototypes.shape[0]
        H = self.num_heads

        # 1. Project Input x into all H spaces
        # x: [B, D], heads: [H, D, D] -> [B, H, D]
        x_proj = torch.einsum("bd,hde->bhe", x, self.metric_heads)

        # 2. Project Prototypes into all H spaces
        # m: [L, D], heads: [H, D, D] -> [L, H, D]
        m_proj = torch.einsum("ld,hde->lhe", self.encoder_prototypes, self.metric_heads)

        x_norm = (x_proj**2).sum(dim=-1).unsqueeze(-1)  # [B, H, 1]
        m_norm = (m_proj**2).sum(dim=-1).t().unsqueeze(0)  # [1, H, L]

        # Cross Term
        # x_proj: [B, H, D], m_proj: [L, H, D]
        # [B, H, D] @ [D, H, L] (permuted m_proj)
        cross = torch.einsum("bhd,lhd->bhl", x_proj, m_proj)

        # Distances per head
        # [B, H, 1] + [1, H, L] - [B, H, L] -> [B, H, L]
        dists_per_head = x_norm + m_norm - 2.0 * cross

        # 3. Weighted Mix
        weights = F.softmax(self.proto_head_logits, dim=1).t().unsqueeze(0)  # [1, H, L]

        # Weighted sum across heads
        # ([B, H, L] * [1, H, L]).sum(dim=1) -> [B, L]
        return (dists_per_head * weights).sum(dim=1)

"""
Compute correlation between latent-space distances and perceptual color distances (HCL/LCH).

This script is intentionally similar in spirit to `visualize_pca.py`, but instead of PCA plots it:

1) Loads activations + metadata from an activations `.pt` file.
2) Optionally loads an SAE model and computes latents (z).
3) Aggregates latents per *color* (derived from filenames like `..._R_G_B.jpg`).
4) Builds:
   - A latent distance matrix between colors (Euclidean by default).
   - A color distance matrix between colors in HCL (via CIELAB/LCH).
5) Reports similarity between the two distance matrices:
   - Pearson and Spearman correlation over upper-triangular entries.
   - Mantel permutation test (optional, but included).
   - Simple rank-based retrieval metrics (nearest-neighbor agreement).
6) Optionally computes *per-color* correlations:
   - For each color i, correlate distances from i to all other colors (latent vs color),
     print those correlations for all colors, and then report the mean across colors.

Notes:
- "HCL" in many visualization contexts corresponds to CIELCh(ab) (L*, C*, h°).
  There are multiple HCL variants (e.g., HCLuv). This implementation uses Lab/LCh(ab)
  with sRGB D65 conversion (common and dependency-free).
- File metadata must include paths whose basenames end with `_{R}_{G}_{B}.jpg`.

Example:
  python -m SpaDE.latent_color_distance_correlation \
    --data_path /path/to/acts.pt \
    --sae_type SzpaDeL \
    --model_path /path/to/model.pth \
    --expansion_factor 4

"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Keep parity with visualize_pca.py import style
sys.path.append(os.path.dirname(__file__))
from models import (  # noqa: E402
    SpaDE,
    TopKSAE,
    ReLUSAE,
    SzpaDeLDiag,
    SzpaDeLDiagLocal,
    SzpaDeL,
    SzpaDeLLocal,
    SzpaDeLRank1,
    SzpaDeLMultiHead,
)


# -----------------------------
# Parsing + model loading
# -----------------------------


def parse_rgb_from_path(path: str) -> Tuple[int, int, int]:
    """
    Extract RGB from filename like 'car_255_0_0.jpg' (or any path ending in _R_G_B.jpg).

    Returns (R, G, B) as ints in [0, 255].
    """
    filename = os.path.basename(path)
    match = re.search(r"_(\d+)_(\d+)_(\d+)\.jpg$", filename)
    if not match:
        raise ValueError(f"Invalid image name (expected _R_G_B.jpg suffix): {filename}")
    r, g, b = map(int, match.groups())
    for v in (r, g, b):
        if v < 0 or v > 255:
            raise ValueError(f"RGB out of range in filename {filename}: {(r, g, b)}")
    return r, g, b


def load_sae_model(
    sae_type: str,
    model_path: str,
    input_dim: int,
    expansion_factor: int,
    k: int = 32,
):
    latent_dim = input_dim * expansion_factor
    models_map = {
        "SpaDE": SpaDE,
        "TopK": lambda i, l: TopKSAE(i, l, k=k),
        "ReLU": ReLUSAE,
        "SzpaDeLDiag": SzpaDeLDiag,
        "SzpaDeLDiagLocal": SzpaDeLDiagLocal,
        "SzpaDeL": SzpaDeL,
        "SzpaDeLLocal": SzpaDeLLocal,
        "SzpaDeLRank1": SzpaDeLRank1,
        "SzpaDeLMultiHead": SzpaDeLMultiHead,
    }

    if sae_type not in models_map:
        raise ValueError(f"Unknown SAE type: {sae_type}")

    if model_path is None:
        raise ValueError("--model_path is required when --sae_type is not Raw")

    model = models_map[sae_type](input_dim, latent_dim)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


# -----------------------------
# Color conversions: sRGB -> Lab -> LCh (HCL)
# -----------------------------


def _srgb_to_linear(u: np.ndarray) -> np.ndarray:
    """
    Convert sRGB channel(s) in [0,1] to linear RGB in [0,1].
    """
    u = np.asarray(u, dtype=np.float64)
    return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)


def _linear_rgb_to_xyz_d65(rgb_lin: np.ndarray) -> np.ndarray:
    """
    Convert linear RGB (sRGB primaries) to XYZ using D65 whitepoint.

    rgb_lin: (..., 3)
    returns: (..., 3) XYZ
    """
    # sRGB -> XYZ (D65) matrix
    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    return rgb_lin @ M.T


def _f_lab(t: np.ndarray) -> np.ndarray:
    """
    CIE Lab f(t) helper.
    """
    delta = 6 / 29
    return np.where(t > delta**3, np.cbrt(t), (t / (3 * delta**2)) + (4 / 29))


def _xyz_d65_to_lab(xyz: np.ndarray) -> np.ndarray:
    """
    XYZ (D65) -> Lab (D65 reference white).

    xyz: (..., 3), with X,Y,Z relative to 1.0 scale (not 100)
    returns: (..., 3) [L*, a*, b*]
    """
    # D65 reference white (sRGB) in XYZ with Y=1.0 scaling
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn

    fx = _f_lab(x)
    fy = _f_lab(y)
    fz = _f_lab(z)

    L = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def rgb255_to_lch(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """
    Convert (R,G,B) in 0..255 to LCh(ab) = (L*, C*, h°).

    Returns:
      L in [0, 100] approx
      C >= 0
      h in degrees [0, 360)
    """
    r, g, b = rgb
    srgb = np.array([r, g, b], dtype=np.float64) / 255.0
    rgb_lin = _srgb_to_linear(srgb)
    xyz = _linear_rgb_to_xyz_d65(rgb_lin)
    lab = _xyz_d65_to_lab(xyz)

    L, a, b2 = float(lab[0]), float(lab[1]), float(lab[2])
    C = math.sqrt(a * a + b2 * b2)
    h = math.degrees(math.atan2(b2, a)) % 360.0
    return L, C, h


def lch_distance(
    lch1: Tuple[float, float, float],
    lch2: Tuple[float, float, float],
    *,
    hue_weight: float = 1.0,
    components: str = "lch",
) -> float:
    """
    Distance between two LCh points.

    Options:
      components="lab": compute Euclidean distance in Lab space (more standard for distances)
      components="lch": compute Euclidean distance in (L, C, hue_angle) with circular hue handling

    For "lch", hue is treated as an angle: we use the shortest angular difference (in degrees),
    and scale it by `hue_weight`.

    Recommendation:
      Use components="lab" for a conventional perceptual-ish distance proxy.
    """
    L1, C1, h1 = lch1
    L2, C2, h2 = lch2

    if components.lower() == "lab":
        # Convert LCh back to Lab for distance: a=C*cos(h), b=C*sin(h).
        h1r = math.radians(h1)
        h2r = math.radians(h2)
        a1, b1 = C1 * math.cos(h1r), C1 * math.sin(h1r)
        a2, b2 = C2 * math.cos(h2r), C2 * math.sin(h2r)
        return math.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)

    if components.lower() != "lch":
        raise ValueError(f"Unknown components mode: {components}")

    # Circular hue difference
    dh = abs(h1 - h2) % 360.0
    dh = min(dh, 360.0 - dh)
    return math.sqrt((L1 - L2) ** 2 + (C1 - C2) ** 2 + (hue_weight * dh) ** 2)


# -----------------------------
# Distance matrices + metrics
# -----------------------------


def pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """
    Compute full pairwise Euclidean distance matrix.

    X: (N, D)
    returns: (N, N)
    """
    X = np.asarray(X, dtype=np.float64)
    # (x - y)^2 = x^2 + y^2 - 2xy
    s = np.sum(X * X, axis=1, keepdims=True)  # (N,1)
    D2 = s + s.T - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)
    return np.sqrt(D2, dtype=np.float64)


def pairwise_cosine_distance(X: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Compute full pairwise cosine distance matrix: d(i,j) = 1 - cos(i,j).

    X: (N, D)
    returns: (N, N)
    """
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, eps)
    # cosine similarity matrix in [-1, 1]
    S = Xn @ Xn.T
    # numerical safety
    S = np.clip(S, -1.0, 1.0)
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    return D


def upper_triangular_values(D: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Extract upper-triangular entries (excluding diagonal by default).
    """
    iu = np.triu_indices(D.shape[0], k=k)
    return D[iu]


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size:
        raise ValueError("pearsonr: x and y must have same length")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return float("nan")
    return float((x @ y) / denom)


def rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """
    Rank data with average ranks for ties (1..n), similar to scipy.stats.rankdata(method="average").
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)

    # handle ties: scan sorted array
    xs = x[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and xs[j] == xs[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0  # average rank positions (1-indexed)
            ranks[order[i:j]] = avg
        i = j
    return ranks


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata_average_ties(x)
    ry = rankdata_average_ties(y)
    return pearsonr(rx, ry)


def per_color_correlations(
    D_lat: np.ndarray,
    D_col: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    For each color i, correlate the vector of distances from i to all j!=i
    between latent-space and color-space.

    Returns:
      per_color_pearson: shape (n,)
      per_color_spearman: shape (n,)
      mean_pearson: scalar (nanmean)
      mean_spearman: scalar (nanmean)
    """
    if D_lat.shape != D_col.shape:
        raise ValueError(f"D_lat and D_col must have same shape, got {D_lat.shape} vs {D_col.shape}")
    if D_lat.ndim != 2 or D_lat.shape[0] != D_lat.shape[1]:
        raise ValueError(f"Distance matrices must be square, got {D_lat.shape}")

    n = D_lat.shape[0]
    per_p = np.full((n,), np.nan, dtype=np.float64)
    per_s = np.full((n,), np.nan, dtype=np.float64)

    for i in range(n):
        mask = np.ones((n,), dtype=bool)
        mask[i] = False

        x = D_lat[i, mask]
        y = D_col[i, mask]

        # Need at least 2 points for correlation. (With n>=3 this is always satisfied,
        # but keep it robust if someone changes filtering upstream.)
        if x.size >= 2:
            per_p[i] = pearsonr(x, y)
            per_s[i] = spearmanr(x, y)

    return per_p, per_s, float(np.nanmean(per_p)), float(np.nanmean(per_s))


def mantel_test(
    A: np.ndarray,
    B: np.ndarray,
    *,
    permutations: int = 1000,
    seed: int = 0,
    method: str = "pearson",
) -> Tuple[float, float]:
    """
    Mantel test between two distance matrices using permutations of labels.

    Returns: (observed_correlation, p_value_two_sided)

    This implementation:
      - Extracts upper triangle of both matrices (excluding diagonal).
      - Computes correlation (pearson or spearman).
      - Permutes rows/cols of B together (same permutation) and recomputes.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("mantel_test: A and B must be same-shape square matrices")

    corr_fn = pearsonr if method.lower() == "pearson" else spearmanr
    a = upper_triangular_values(A, k=1)
    b = upper_triangular_values(B, k=1)
    obs = corr_fn(a, b)

    rng = np.random.default_rng(seed)
    more_extreme = 0
    for _ in range(permutations):
        p = rng.permutation(A.shape[0])
        Bp = B[p][:, p]
        bp = upper_triangular_values(Bp, k=1)
        c = corr_fn(a, bp)
        if not np.isfinite(c):
            continue
        if abs(c) >= abs(obs):
            more_extreme += 1
    # add-one smoothing
    p_value = (more_extreme + 1) / (permutations + 1)
    return float(obs), float(p_value)


def nearest_neighbor_agreement(
    A: np.ndarray,
    B: np.ndarray,
    *,
    top_k: int = 1,
) -> float:
    """
    For each row i, compare the set of top_k nearest neighbors (excluding i) induced by A vs B.

    Returns mean Jaccard similarity between neighbor sets.
    """
    n = A.shape[0]
    if n <= 1:
        return float("nan")

    def topk_neighbors(D: np.ndarray, i: int) -> np.ndarray:
        # exclude self by setting inf
        row = D[i].copy()
        row[i] = np.inf
        # argsort then take first k
        return np.argsort(row)[:top_k]

    scores = []
    for i in range(n):
        na = set(map(int, topk_neighbors(A, i)))
        nb = set(map(int, topk_neighbors(B, i)))
        inter = len(na & nb)
        union = len(na | nb)
        scores.append(inter / union if union else 0.0)
    return float(np.mean(scores))


@dataclass(frozen=True)
class ColorGroup:
    rgb: Tuple[int, int, int]
    lch: Tuple[float, float, float]
    latent_mean: np.ndarray
    count: int


def aggregate_latents_by_color(
    features: np.ndarray,
    metadata: Sequence[str],
    *,
    patches_per_img: Optional[int] = None,
) -> List[ColorGroup]:
    """
    Map each activation row to an RGB color derived from metadata file paths,
    then aggregate by unique color using mean latent.

    patches_per_img:
      If provided, repeats each metadata color this many times to align with features rows.
      If None, will infer similarly to visualize_pca.py:
        - if num_acts > num_meta, assumes constant patches_per_img = num_acts // num_meta
        - else 1:1 mapping
    """
    features = np.asarray(features)
    num_acts = features.shape[0]
    num_meta = len(metadata)

    if patches_per_img is None:
        if num_acts > num_meta:
            patches_per_img = num_acts // num_meta
        else:
            patches_per_img = 1

    # Build per-row RGBs
    rgbs: List[Tuple[int, int, int]] = []
    if patches_per_img == 1 and num_acts == num_meta:
        for p in metadata:
            rgbs.append(parse_rgb_from_path(p))
    else:
        # Repeat each image's color patches_per_img times, then truncate to num_acts
        for p in metadata:
            rgb = parse_rgb_from_path(p)
            rgbs.extend([rgb] * patches_per_img)
        rgbs = rgbs[:num_acts]

    if len(rgbs) != num_acts:
        raise RuntimeError(
            f"Color mapping mismatch: got {len(rgbs)} colors for {num_acts} activations "
            f"(patches_per_img={patches_per_img}, num_meta={num_meta})."
        )

    # Aggregate
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for i, rgb in enumerate(rgbs):
        buckets.setdefault(rgb, []).append(i)

    groups: List[ColorGroup] = []
    for rgb, idxs in sorted(buckets.items(), key=lambda kv: kv[0]):
        idxs_arr = np.array(idxs, dtype=np.int64)
        latent_mean = features[idxs_arr].mean(axis=0)
        lch = rgb255_to_lch(rgb)
        groups.append(ColorGroup(rgb=rgb, lch=lch, latent_mean=latent_mean, count=len(idxs)))

    return groups


def build_color_distance_matrix(
    groups: Sequence[ColorGroup],
    *,
    color_distance: str = "lab",
    hue_weight: float = 1.0,
) -> np.ndarray:
    """
    Build a pairwise distance matrix between colors.

    color_distance:
      - "lab": Euclidean distance in Lab (derived from LCh)
      - "lch": Euclidean-like distance in (L, C, h°) with circular hue handling (uses hue_weight)
      - "lch_euclidean": plain Euclidean distance on raw LCh vectors [L, C, h] (NO circular hue handling)
    """
    n = len(groups)
    cd = color_distance.lower()

    if cd == "lch_euclidean":
        X = np.stack([np.array([g.lch[0], g.lch[1], g.lch[2]], dtype=np.float64) for g in groups], axis=0)
        return pairwise_euclidean(X)

    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = lch_distance(
                groups[i].lch,
                groups[j].lch,
                components=color_distance,
                hue_weight=hue_weight,
            )
            D[i, j] = D[j, i] = d
    return D


def normalize_vector(x: np.ndarray) -> np.ndarray:
    """
    Z-score normalize a vector. Returns zeros if std is 0.
    """
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    sd = x.std()
    if sd == 0:
        return np.zeros_like(x)
    return (x - mu) / sd


def format_rgb(rgb: Tuple[int, int, int]) -> str:
    return f"{rgb[0]:3d},{rgb[1]:3d},{rgb[2]:3d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute correlation between latent distance matrix and HCL(LCh) color distance matrix."
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to activations .pt file")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained SAE .pth model (omit for Raw)",
    )
    parser.add_argument(
        "--sae_type",
        type=str,
        choices=[
            "Raw",
            "ReLU",
            "TopK",
            "SpaDE",
            "SzpaDeLDiag",
            "SzpaDeLDiagLocal",
            "SzpaDeL",
            "SzpaDeLLocal",
            "SzpaDeLRank1",
            "SzpaDeLMultiHead",
        ],
        default="Raw",
    )
    parser.add_argument("--expansion_factor", type=int, default=4)
    parser.add_argument("--k", type=int, default=32, help="k for TopK SAE")

    parser.add_argument(
        "--latent_distance",
        type=str,
        choices=["euclidean", "cosine"],
        default="euclidean",
        help="Distance metric in latent space.",
    )
    parser.add_argument(
        "--color_distance",
        type=str,
        choices=["lab", "lch", "lch_euclidean"],
        default="lab",
        help="Distance metric in color space using HCL/LCh representation.",
    )
    parser.add_argument(
        "--hue_weight",
        type=float,
        default=1.0,
        help="Only used when --color_distance=lch: scale for hue (degrees).",
    )

    parser.add_argument(
        "--patches_per_img",
        type=int,
        default=None,
        help="Override patches per image mapping (otherwise inferred like visualize_pca.py).",
    )

    parser.add_argument(
        "--min_count_per_color",
        type=int,
        default=1,
        help="Filter out colors that appear fewer than this many times.",
    )

    parser.add_argument(
        "--mantel_permutations",
        type=int,
        default=0,
        help="If >0, run Mantel permutation test with this many permutations.",
    )
    parser.add_argument(
        "--mantel_seed",
        type=int,
        default=0,
        help="Random seed for Mantel permutations.",
    )

    parser.add_argument(
        "--topk_agreement",
        type=int,
        default=1,
        help="Compute nearest-neighbor agreement with this k (Jaccard over top-k neighbors).",
    )

    parser.add_argument(
        "--per_color_correlations",
        action="store_true",
        help=(
            "Compute per-color correlations: for each color i, correlate the vector of "
            "distances from i to all other colors (latent vs color). Prints all per-color "
            "values and then reports the mean."
        ),
    )

    args = parser.parse_args()

    print(f"Loading data from {args.data_path}...")
    data = torch.load(args.data_path, map_location="cpu")

    activations = data["activations"].to(torch.float32)
    metadata = data["metadata"]
    cat_tag = data.get("category", "unknown")

    raw_features = activations.cpu().numpy()

    if args.sae_type == "Raw":
        features = raw_features
        model_tag = "Raw"
        print("Using Raw activations as features.")
    else:
        print(f"Loading model: {args.sae_type} from {args.model_path} ...")
        model = load_sae_model(
            args.sae_type,
            args.model_path,
            activations.shape[-1],
            args.expansion_factor,
            args.k,
        )
        print(f"Computing latent activations (z) using {args.sae_type} SAE...")
        with torch.no_grad():
            _, latents = model(activations)
        features = latents.cpu().numpy()
        model_tag = args.sae_type

    groups = aggregate_latents_by_color(
        features,
        metadata,
        patches_per_img=args.patches_per_img,
    )

    if args.min_count_per_color > 1:
        groups = [g for g in groups if g.count >= args.min_count_per_color]

    if len(groups) < 3:
        raise SystemExit(
            f"Need at least 3 unique colors after filtering; got {len(groups)}. "
            f"(Try lowering --min_count_per_color)"
        )

    print(f"Category: {cat_tag}")
    print(f"Model: {model_tag}")
    print(f"Unique colors: {len(groups)}")
    print(
        "Counts per color (first 10): "
        + ", ".join([f"[{format_rgb(g.rgb)}]={g.count}" for g in groups[:10]])
        + (" ..." if len(groups) > 10 else "")
    )

    latent_means = np.stack([g.latent_mean for g in groups], axis=0)

    # Latent distance matrix
    if args.latent_distance == "euclidean":
        D_lat = pairwise_euclidean(latent_means)
    elif args.latent_distance == "cosine":
        D_lat = pairwise_cosine_distance(latent_means)
    else:
        raise ValueError(f"Unsupported latent distance: {args.latent_distance}")

    # Color distance matrix (in HCL/LCh)
    D_col = build_color_distance_matrix(
        groups,
        color_distance=args.color_distance,
        hue_weight=args.hue_weight,
    )

    # Vectorize and compare
    v_lat = upper_triangular_values(D_lat, k=1)
    v_col = upper_triangular_values(D_col, k=1)

    pear = pearsonr(v_lat, v_col)
    spear = spearmanr(v_lat, v_col)

    # Also compare after z-scoring (doesn't change Pearson, but good for reporting sanity)
    v_lat_z = normalize_vector(v_lat)
    v_col_z = normalize_vector(v_col)
    pear_z = pearsonr(v_lat_z, v_col_z)
    spear_z = spearmanr(v_lat_z, v_col_z)

    # Nearest-neighbor agreement
    nna = nearest_neighbor_agreement(D_lat, D_col, top_k=args.topk_agreement)

    print("\n=== Distance matrix similarity ===")
    print(f"Latent distance: {args.latent_distance} over mean latent per color")
    print(f"Color distance:  {args.color_distance} via sRGB->Lab->LCh (HCL)")
    if args.color_distance == "lch":
        print(f"Hue weight:     {args.hue_weight}")

    print("\nVectorized upper-triangle comparisons:")
    print(f"Pearson r:      {pear:.6f} (z-scored: {pear_z:.6f})")
    print(f"Spearman rho:   {spear:.6f} (z-scored: {spear_z:.6f})")
    print(f"NN agreement:   {nna:.6f} (top_k={args.topk_agreement})")

    if args.per_color_correlations:
        per_p, per_s, mean_p, mean_s = per_color_correlations(D_lat, D_col)

        print("\nPer-color correlations (each color i vs all other colors):")
        print("  idx  rgb           count    Pearson_r    Spearman_rho")
        for i, g in enumerate(groups):
            p_i = per_p[i]
            s_i = per_s[i]
            print(
                f"  {i:3d}  ({format_rgb(g.rgb)})  {g.count:6d}  "
                f"{p_i:10.6f}  {s_i:12.6f}"
            )

        print("\nPer-color correlation means (nanmean over colors):")
        print(f"Mean Pearson r:    {mean_p:.6f}")
        print(f"Mean Spearman rho: {mean_s:.6f}")

    if args.mantel_permutations and args.mantel_permutations > 0:
        print("\nMantel test (two-sided):")
        obs_p, pval_p = mantel_test(
            D_lat,
            D_col,
            permutations=args.mantel_permutations,
            seed=args.mantel_seed,
            method="pearson",
        )
        obs_s, pval_s = mantel_test(
            D_lat,
            D_col,
            permutations=args.mantel_permutations,
            seed=args.mantel_seed,
            method="spearman",
        )
        print(
            f"Pearson Mantel r:  {obs_p:.6f}, p={pval_p:.6f} "
            f"(permutations={args.mantel_permutations})"
        )
        print(
            f"Spearman Mantel r: {obs_s:.6f}, p={pval_s:.6f} "
            f"(permutations={args.mantel_permutations})"
        )

    # Optional: show a small excerpt of matrices for sanity
    print("\nSanity check (first 5 colors):")
    for i, g in enumerate(groups[:5]):
        L, C, h = g.lch
        print(
            f"  idx={i:2d} rgb=({format_rgb(g.rgb)}) "
            f"LCh=({L:6.2f},{C:6.2f},{h:7.2f}°) count={g.count}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

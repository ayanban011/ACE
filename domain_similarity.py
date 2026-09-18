"""
Metrics: FID, CKA (sample-matched), EMD
"""

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg
from tqdm import tqdm
import ot
import seaborn as sns
import matplotlib.pyplot as plt
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ----------------------------
# 1. Feature extraction
# ----------------------------
def extract_features(folder, model, transform, device, batch_size=64, max_samples=None):
    dataset = datasets.ImageFolder(folder, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc=f"Extracting {os.path.basename(folder)}"):
            imgs = imgs.to(device)
            f = model(imgs)
            f = f.view(f.size(0), -1)  # ensures 2D shape even for batch=1
            feats.append(f.cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    if max_samples and len(feats) > max_samples:
        idx = np.random.choice(len(feats), max_samples, replace=False)
        feats = feats[idx]
    return feats


def build_feature_extractor():
    base = models.resnet18(pretrained=True)
    model = nn.Sequential(*list(base.children())[:-1])  # remove FC layer
    model.eval()
    return model

# ----------------------------
# 2. Helper: match sample counts
# ----------------------------
def match_sample_count(X, Y, seed=42):
    np.random.seed(seed)
    n = min(len(X), len(Y))
    idx_X = np.random.choice(len(X), n, replace=False)
    idx_Y = np.random.choice(len(Y), n, replace=False)
    return X[idx_X], Y[idx_Y]

# ----------------------------
# 3. Metrics
# ----------------------------
def compute_fid(x, y):
    mu1, sigma1 = x.mean(0), np.cov(x, rowvar=False)
    mu2, sigma2 = y.mean(0), np.cov(y, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def linear_cka(X, Y):
    X -= X.mean(0, keepdims=True)
    Y -= Y.mean(0, keepdims=True)
    XTX = X @ X.T
    YTY = Y @ Y.T
    hsic = np.sum(XTX * YTY)
    var1 = np.sqrt(np.sum(XTX * XTX))
    var2 = np.sqrt(np.sum(YTY * YTY))
    return hsic / (var1 * var2 + 1e-12)

def earth_movers_distance(X, Y, normalize=True):
    if normalize:
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    cost = ot.dist(X, Y, metric="euclidean")
    cost /= cost.max()
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    emd2 = ot.emd2(a, b, cost)
    return np.sqrt(emd2)

# ----------------------------
# 4. Main computation
# ----------------------------
def compute_domain_similarities(domain_dirs, max_samples=2000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_feature_extractor().to(device)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    # Extract features per domain
    domain_feats = {d: extract_features(path, model, transform, device, max_samples=max_samples)
                    for d, path in domain_dirs.items()}

    domains = list(domain_feats.keys())
    n = len(domains)
    fid_mat = np.zeros((n, n))
    cka_mat = np.zeros((n, n))
    emd_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x, y = domain_feats[domains[i]], domain_feats[domains[j]]
            # Ensure same sample size for CKA
            x_matched, y_matched = match_sample_count(x, y)
            fid_mat[i,j] = compute_fid(x, y)
            cka_mat[i,j] = linear_cka(x_matched, y_matched)
            emd_mat[i,j] = earth_movers_distance(x, y)

    return domains, fid_mat, cka_mat, emd_mat

# ----------------------------
# 5. Visualization
# ----------------------------
def plot_heatmap(matrix, domains, title, cmap, fmt=".2f"):
    plt.figure(figsize=(6,5))
    sns.heatmap(matrix, annot=True, xticklabels=domains, yticklabels=domains,
                cmap=cmap, fmt=fmt, square=True, cbar_kws={'label': title})
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ----------------------------
# 5. Example usage
# ----------------------------
if __name__ == "__main__":
    domain_dirs = {
        "clipart": "DomainNet/clipart",
        "infograph": "DomainNet/infograph",
        "painting": "DomainNet/painting",
        "quickdraw": "DomainNet/quickdraw",
        "real": "DomainNet/real",
        "sketch": "DomainNet/sketch"
    }

    domains, fid, cka, emd = compute_domain_similarities(domain_dirs)

    print("\nFID matrix (lower = more similar):\n", fid)
    print("\nCKA matrix (higher = more similar):\n", cka)
    print("\nEMD matrix (lower = more similar):\n", emd)

    # Visualize as heatmaps
    plot_heatmap(fid, domains, "Fréchet Inception Distance (FID)", cmap="mako_r")
    plot_heatmap(cka, domains, "Centered Kernel Alignment (CKA)", cmap="crest")
    plot_heatmap(emd, domains, "Earth Mover’s Distance (EMD)", cmap="magma_r")

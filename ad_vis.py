import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # Set torch cache directory

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

def plot_tsne_with_boundary(domain_features, domain1, domain2):
    feats1 = domain_features[domain1]
    feats2 = domain_features[domain2]

    X = np.vstack([feats1, feats2])
    y = np.array([0]*len(feats1) + [1]*len(feats2))

    # t-SNE
    from sklearn.manifold import TSNE
    X_emb = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

    # Train classifier
    clf = LogisticRegression().fit(X_emb, y)

    # Mesh grid
    x_min, x_max = X_emb[:, 0].min() - 1, X_emb[:, 0].max() + 1
    y_min, y_max = X_emb[:, 1].min() - 1, X_emb[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, alpha=0.2)

    plt.scatter(X_emb[:len(feats1), 0], X_emb[:len(feats1), 1],
                label=domain1, alpha=0.6)
    plt.scatter(X_emb[len(feats1):, 0], X_emb[len(feats1):, 1],
                label=domain2, alpha=0.6)

    plt.legend()
    plt.title(f"Decision Boundary: {domain1} vs {domain2}")
    plt.savefig(f"Decision Boundary{domain1}vs{domain2}.png")

from sklearn.model_selection import cross_val_score

def overlap_score_linear(X_emb, y):
    clf = LogisticRegression(max_iter=1000)
    acc = cross_val_score(clf, X_emb, y, cv=5).mean()
    return acc

def bhattacharyya_distance(X1, X2):
    mu1, mu2 = X1.mean(0), X2.mean(0)
    cov1, cov2 = np.cov(X1.T), np.cov(X2.T)
    cov_avg = (cov1 + cov2) / 2

    term1 = 0.125 * (mu1 - mu2).T @ np.linalg.inv(cov_avg) @ (mu1 - mu2)
    term2 = 0.5 * np.log(
        np.linalg.det(cov_avg) /
        np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2))
    )
    return term1 + term2

from sklearn.neighbors import NearestNeighbors

def knn_overlap(X_emb, y, k=10):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_emb)
    _, indices = nbrs.kneighbors(X_emb)

    same_domain = 0
    total = 0

    for i in range(len(X_emb)):
        for j in indices[i]:
            if y[i] == y[j]:
                same_domain += 1
            total += 1

    return 1 - (same_domain / total)  # higher = more overlap

def compute_overlap_metrics(domain_features, domain1, domain2):
    feats1 = domain_features[domain1]
    feats2 = domain_features[domain2]

    X = np.vstack([feats1, feats2])
    y = np.array([0]*len(feats1) + [1]*len(feats2))

    from sklearn.manifold import TSNE
    X_emb = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

    acc = overlap_score_linear(X_emb, y)
    bhat = bhattacharyya_distance(X_emb[:len(feats1)], X_emb[len(feats1):])
    knn = knn_overlap(X_emb, y)

    print(f"{domain1} vs {domain2}")
    print(f"Linear separability (↑=less overlap): {acc:.3f}")
    print(f"Bhattacharyya distance (↑=less overlap): {bhat:.3f}")
    print(f"kNN overlap (↑=more overlap): {knn:.3f}")

import matplotlib.animation as animation

def animate_domain_shift(domain_features, domain1, domain2):
    feats1 = domain_features[domain1]
    feats2 = domain_features[domain2]

    n = min(len(feats1), len(feats2))
    feats1, feats2 = feats1[:n], feats2[:n]

    fig, ax = plt.subplots(figsize=(6, 5))

    def update(t):
        ax.clear()

        alpha = t / 20.0
        interp = (1 - alpha) * feats1 + alpha * feats2

        X = np.vstack([feats1, interp, feats2])

        from sklearn.manifold import TSNE
        emb = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

        ax.scatter(emb[:n, 0], emb[:n, 1], label=domain1, alpha=0.4)
        ax.scatter(emb[n:2*n, 0], emb[n:2*n, 1], label="interpolation", alpha=0.6)
        ax.scatter(emb[2*n:, 0], emb[2*n:, 1], label=domain2, alpha=0.4)

        ax.set_title(f"Domain Shift {domain1} → {domain2} (α={alpha:.2f})")
        ax.legend()

    ani = animation.FuncAnimation(fig, update, frames=20, repeat=True)
    plt.show()

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"

# PACS domains
domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

# Image transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load pretrained model (feature extractor)
model = models.resnet50(pretrained=True)
model.fc = nn.Identity()  # remove classification head
model = model.to(device)
model.eval()

def load_domain(root, domain):
    dataset = datasets.ImageFolder(root=f"{root}/{domain}", transform=transform)
    return DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True)

data_root = "./DomainNet"

loaders = {d: load_domain(data_root, d) for d in domains}

def extract_features(loader):
    features = []
    
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            f = model(x)
            features.append(f.cpu().numpy())
    
    return np.vstack(features)

domain_features = {}

for d in domains:
    print(f"Extracting features for {d}")
    domain_features[d] = extract_features(loaders[d])

# Decision boundary
plot_tsne_with_boundary(domain_features, "clipart", "infograph")
plot_tsne_with_boundary(domain_features, "clipart", "painting")
plot_tsne_with_boundary(domain_features, "clipart", "quickdraw")
plot_tsne_with_boundary(domain_features, "clipart", "real")
plot_tsne_with_boundary(domain_features, "clipart", "sketch")
plot_tsne_with_boundary(domain_features, "infograph", "painting")
plot_tsne_with_boundary(domain_features, "infograph", "quickdraw")
plot_tsne_with_boundary(domain_features, "infograph", "real")
plot_tsne_with_boundary(domain_features, "infograph", "sketch")
plot_tsne_with_boundary(domain_features, "painting", "quickdraw")
plot_tsne_with_boundary(domain_features, "painting", "real")
plot_tsne_with_boundary(domain_features, "painting", "sketch")
plot_tsne_with_boundary(domain_features, "quickdraw", "real")
plot_tsne_with_boundary(domain_features, "quickdraw", "sketch")
plot_tsne_with_boundary(domain_features, "real", "sketch")

# Decision boundary
compute_overlap_metrics(domain_features, "clipart", "infograph")
compute_overlap_metrics(domain_features, "clipart", "painting")
compute_overlap_metrics(domain_features, "clipart", "quickdraw")
compute_overlap_metrics(domain_features, "clipart", "real")
compute_overlap_metrics(domain_features, "clipart", "sketch")
compute_overlap_metrics(domain_features, "infograph", "painting")
compute_overlap_metrics(domain_features, "infograph", "quickdraw")
compute_overlap_metrics(domain_features, "infograph", "real")
compute_overlap_metrics(domain_features, "infograph", "sketch")
compute_overlap_metrics(domain_features, "painting", "quickdraw")
compute_overlap_metrics(domain_features, "painting", "real")
compute_overlap_metrics(domain_features, "painting", "sketch")
compute_overlap_metrics(domain_features, "quickdraw", "real")
compute_overlap_metrics(domain_features, "quickdraw", "sketch")
compute_overlap_metrics(domain_features, "real", "sketch")

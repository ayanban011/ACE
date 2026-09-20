"""
Usage:
    python run_experiments.py --data_root /path/to/PACS --output_dir ./results
"""

import os
import sys
import json
import argparse
import itertools
import warnings
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from scipy.stats import spearmanr, bootstrap
from scipy.spatial.distance import cosine as cosine_dist
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 0. Configuration
# ============================================================

PACS_DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]
PACS_CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
NUM_CLASSES = len(PACS_CLASSES)

ARCHITECTURES = ["resnet18", "resnet50", "vit_small", "mlp_mixer"]
OBJECTIVES = ["erm", "irm", "coral", "dann"]
REGULARIZATIONS = ["none", "dropout", "weight_decay"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. CAS-H Computation (from similarity matrix S)
# ============================================================


def compute_cas(S: np.ndarray) -> float:
    """
    Compute Circuit Alignment Score from similarity matrix S.
    CAS = (1/c) * sum_i S_ii  -  (1/(c*(c-1))) * sum_{i!=j} S_ij

    Args:
        S: (c x c) class-aligned similarity matrix where S_ij = K(C1^i, C2^j)

    Returns:
        CAS score (float)
    """
    c = S.shape[0]
    assert S.shape == (c, c), f"Expected square matrix, got {S.shape}"

    diagonal_coherence = np.trace(S) / c
    off_diagonal_mask = ~np.eye(c, dtype=bool)
    off_diagonal_confusion = S[off_diagonal_mask].sum() / (c * (c - 1))

    return float(diagonal_coherence - off_diagonal_confusion)


def compute_cas_h(S: np.ndarray, alpha: float = 10.0) -> float:
    """
    Compute CAS-H (entropy-based variant) from similarity matrix S.

    CAS-H = (1 / (c * log(c))) * sum_i H(P_i)

    where P_i(j) = exp(alpha * S_ij) / sum_k exp(alpha * S_ik)  (softmax)
    and H(P_i) = -sum_j P_i(j) * log(P_i(j))                   (Shannon entropy)

    CAS-H in [0, 1]:
      0 = perfect alignment (each P_i is delta on diagonal)
      1 = maximal confusion (each P_i is uniform)

    Args:
        S: (c x c) class-aligned similarity matrix
        alpha: softmax temperature (higher = sharper)

    Returns:
        CAS-H score (float)
    """
    c = S.shape[0]
    assert S.shape == (c, c)
    assert c > 1, "Need at least 2 classes for CAS-H"

    # Row-wise softmax with temperature alpha
    # Use log-sum-exp trick for numerical stability
    logits = alpha * S  # (c, c)
    logits_max = logits.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(logits - logits_max).sum(axis=1, keepdims=True)) + logits_max
    log_P = logits - log_norm  # (c, c) log-probabilities
    P = np.exp(log_P)  # (c, c) probabilities

    # Shannon entropy per row
    # H(P_i) = -sum_j P_i(j) * log(P_i(j))
    # Use log_P directly for numerical stability
    entropies = np.zeros(c)
    for i in range(c):
        # Avoid log(0) by masking near-zero probabilities
        mask = P[i] > 1e-12
        entropies[i] = -np.sum(P[i, mask] * log_P[i, mask])

    # Normalize: mean entropy / log(c)
    cas_h = entropies.mean() / np.log(c)

    return float(np.clip(cas_h, 0.0, 1.0))


def compute_row_entropies(S: np.ndarray, alpha: float = 10.0) -> np.ndarray:
    """
    Compute per-class row entropies H(P_i) from S.
    Used for class-level vulnerability analysis and Theorem 4.

    Args:
        S: (c x c) similarity matrix
        alpha: softmax temperature

    Returns:
        entropies: (c,) array of per-class entropies
    """
    c = S.shape[0]
    logits = alpha * S
    logits_max = logits.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(logits - logits_max).sum(axis=1, keepdims=True)) + logits_max
    log_P = logits - log_norm
    P = np.exp(log_P)

    entropies = np.zeros(c)
    for i in range(c):
        mask = P[i] > 1e-12
        entropies[i] = -np.sum(P[i, mask] * log_P[i, mask])

    return entropies


def compute_cross_domain_cas(
    circuit_families: Dict[str, List],
    graph_kernel_fn,
    domains: List[str],
) -> Tuple[float, float]:
    """
    Compute cross-domain CAS and CAS-H for a learner across training domains.

    overline{CAS}(f) = (2 / (K*(K-1))) * sum_{i<j} CAS(C(f,d_i), C(f,d_j))

    Args:
        circuit_families: dict mapping domain_name -> list of c circuit graphs
        graph_kernel_fn: callable(C1_graph, C2_graph) -> float
        domains: list of training domain names

    Returns:
        (cross_domain_cas, cross_domain_cas_h) tuple
    """
    K = len(domains)
    assert K >= 2, "Need at least 2 training domains"

    cas_values = []
    cas_h_values = []

    for i in range(K):
        for j in range(i + 1, K):
            d_i, d_j = domains[i], domains[j]
            circuits_i = circuit_families[d_i]  # list of c graphs
            circuits_j = circuit_families[d_j]  # list of c graphs
            c = len(circuits_i)
            assert len(circuits_j) == c

            # Build similarity matrix S
            S = np.zeros((c, c))
            for ci in range(c):
                for cj in range(c):
                    S[ci, cj] = graph_kernel_fn(circuits_i[ci], circuits_j[cj])

            cas_values.append(compute_cas(S))
            cas_h_values.append(compute_cas_h(S))

    num_pairs = K * (K - 1) / 2
    cross_cas = np.mean(cas_values)
    cross_cas_h = np.mean(cas_h_values)

    return float(cross_cas), float(cross_cas_h)


# ============================================================
# 2. PACS Dataset Setup
# ============================================================


def get_pacs_transforms(train: bool = True):
    """Standard PACS transforms following DomainBed."""
    if train:
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.3, 0.3, 0.3, 0.3),
            T.RandomGrayscale(p=0.1),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def load_pacs_domain(data_root: str, domain: str, train: bool = True):
    """
    Load a single PACS domain as an ImageFolder dataset.
    Expected structure: data_root/PACS/{domain}/class_name/*.jpg
    """
    domain_path = os.path.join(data_root, "PACS", domain)
    if not os.path.isdir(domain_path):
        # Try alternate structure: data_root/{domain}/
        domain_path = os.path.join(data_root, domain)
    if not os.path.isdir(domain_path):
        raise FileNotFoundError(
            f"PACS domain '{domain}' not found at {domain_path}. "
            f"Expected structure: {data_root}/PACS/{domain}/class_name/*.jpg"
        )
    transform = get_pacs_transforms(train=train)
    return datasets.ImageFolder(domain_path, transform=transform)


def get_class_indices(dataset, class_idx: int) -> List[int]:
    """Get indices of samples belonging to a specific class."""
    return [i for i, (_, y) in enumerate(dataset.samples) if y == class_idx]


# ============================================================
# 3. Adapter Module
# ============================================================


class MLPAdapter(nn.Module):
    """
    Lightweight two-layer MLP adapter.
    A(z) = W_up * sigma(W_down * z)
    With residual: h = A(z) + z
    """

    def __init__(self, dim: int, bottleneck_ratio: int = 4):
        super().__init__()
        hidden = dim // bottleneck_ratio
        self.down = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(hidden, dim, bias=False)
        self.act = nn.ReLU()

        # Initialize near-zero so adapter starts as identity
        nn.init.zeros_(self.up.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(z))) + z


class AdapterWrappedModel(nn.Module):
    """
    Frozen backbone + trainable adapters + trainable head.
    Architecture-agnostic: supports ResNet, ViT, MLP-Mixer.
    """

    def __init__(
        self,
        arch: str,
        num_classes: int = 7,
        bottleneck_ratio: int = 4,
        num_adapter_layers: int = 4,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.arch = arch
        self.num_adapter_layers = num_adapter_layers

        # ---- Build backbone (frozen) ----
        if arch == "resnet18":
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.feat_dim = 512
            # Extract layer groups for adapter insertion
            self.backbone_blocks = nn.ModuleList([
                nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool),
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
            ])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self._arch_type = "cnn"
            adapter_dims = [64, 64, 128, 256, 512]

        elif arch == "resnet50":
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.feat_dim = 2048
            self.backbone_blocks = nn.ModuleList([
                nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool),
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
            ])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self._arch_type = "cnn"
            adapter_dims = [64, 256, 512, 1024, 2048]

        elif arch == "vit_small":
            try:
                import timm
                backbone = timm.create_model("vit_small_patch16_224", pretrained=True)
            except ImportError:
                raise ImportError("Install timm: pip install timm")
            self.feat_dim = 384
            self.backbone_blocks = nn.ModuleList()
            # Split transformer blocks into groups
            blocks = list(backbone.blocks)
            n = len(blocks)
            group_size = max(1, n // num_adapter_layers)
            for start in range(0, n, group_size):
                end = min(start + group_size, n)
                self.backbone_blocks.append(nn.Sequential(*blocks[start:end]))
            self.patch_embed = backbone.patch_embed
            self.cls_token = backbone.cls_token
            self.pos_embed = backbone.pos_embed
            self.pos_drop = backbone.pos_drop
            self.norm = backbone.norm
            self.pool = None
            self._arch_type = "vit"
            adapter_dims = [384] * len(self.backbone_blocks)

        elif arch == "mlp_mixer":
            try:
                import timm
                backbone = timm.create_model("mixer_b16_224", pretrained=True)
            except ImportError:
                raise ImportError("Install timm: pip install timm")
            self.feat_dim = 768
            self.backbone_blocks = nn.ModuleList()
            blocks = list(backbone.blocks)
            n = len(blocks)
            group_size = max(1, n // num_adapter_layers)
            for start in range(0, n, group_size):
                end = min(start + group_size, n)
                self.backbone_blocks.append(nn.Sequential(*blocks[start:end]))
            self.stem = backbone.stem
            self.norm = backbone.norm
            self.pool = None
            self._arch_type = "mixer"
            adapter_dims = [768] * len(self.backbone_blocks)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        # Freeze backbone
        for block in self.backbone_blocks:
            for p in block.parameters():
                p.requires_grad = False
        if hasattr(self, "patch_embed"):
            for p in self.patch_embed.parameters():
                p.requires_grad = False
        if hasattr(self, "stem"):
            for p in self.stem.parameters():
                p.requires_grad = False

        # ---- Adapters (trainable) ----
        # Insert adapters after the last `num_adapter_layers` blocks
        num_blocks = len(self.backbone_blocks)
        adapter_insertion_indices = list(
            range(max(0, num_blocks - num_adapter_layers), num_blocks)
        )
        self.adapter_indices = adapter_insertion_indices

        self.adapters = nn.ModuleDict()
        for idx in adapter_insertion_indices:
            dim = adapter_dims[idx]
            self.adapters[str(idx)] = MLPAdapter(dim, bottleneck_ratio)

        # ---- Classification head (trainable) ----
        head_layers = [nn.Linear(self.feat_dim, num_classes)]
        if dropout_rate > 0:
            head_layers.insert(0, nn.Dropout(dropout_rate))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Intermediate representations for circuit extraction
        self._intermediates = {}

        if self._arch_type == "cnn":
            z = x
            for i, block in enumerate(self.backbone_blocks):
                z = block(z)
                if str(i) in self.adapters:
                    B, C, H, W = z.shape
                    z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)
                    z_flat = self.adapters[str(i)](z_flat)
                    z = z_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
                self._intermediates[i] = z.detach()
            z = self.pool(z).flatten(1)

        elif self._arch_type == "vit":
            z = self.patch_embed(x)
            cls_tokens = self.cls_token.expand(z.shape[0], -1, -1)
            z = torch.cat([cls_tokens, z], dim=1)
            z = z + self.pos_embed
            z = self.pos_drop(z)
            for i, block in enumerate(self.backbone_blocks):
                z = block(z)
                if str(i) in self.adapters:
                    z = self.adapters[str(i)](z)
                self._intermediates[i] = z.detach()
            z = self.norm(z)
            z = z[:, 0]  # CLS token

        elif self._arch_type == "mixer":
            z = self.stem(x)
            for i, block in enumerate(self.backbone_blocks):
                z = block(z)
                if str(i) in self.adapters:
                    z = self.adapters[str(i)](z)
                self._intermediates[i] = z.detach()
            z = self.norm(z)
            z = z.mean(dim=1)  # Global average pool over patches

        logits = self.head(z)
        return logits

    def get_trainable_params(self):
        """Return only adapter + head parameters for optimization."""
        params = []
        for adapter in self.adapters.values():
            params.extend(adapter.parameters())
        params.extend(self.head.parameters())
        return params


# ============================================================
# 4. Training Objectives
# ============================================================


def erm_loss(logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
    """Standard empirical risk minimization."""
    return F.cross_entropy(logits, labels)


def irm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    domain_labels: torch.Tensor,
    penalty_weight: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """
    Invariant Risk Minimization (IRMv1 penalty).
    L = sum_d CE(f(x_d), y_d) + lambda * sum_d || grad(CE(w*f(x_d), y_d)) ||^2
    where w=1.0 is a scalar dummy classifier.
    """
    domains = domain_labels.unique()
    total_loss = torch.tensor(0.0, device=logits.device)
    penalty = torch.tensor(0.0, device=logits.device)

    for d in domains:
        mask = domain_labels == d
        if mask.sum() == 0:
            continue
        d_logits = logits[mask]
        d_labels = labels[mask]

        # ERM loss for this domain
        d_loss = F.cross_entropy(d_logits, d_labels)
        total_loss = total_loss + d_loss

        # IRM penalty: gradient of loss w.r.t. dummy scalar classifier
        scale = torch.ones(1, device=logits.device, requires_grad=True)
        d_loss_scaled = F.cross_entropy(d_logits * scale, d_labels)
        grad = torch.autograd.grad(d_loss_scaled, scale, create_graph=True)[0]
        penalty = penalty + grad.pow(2).sum()

    return total_loss / len(domains) + penalty_weight * penalty


def coral_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    features: torch.Tensor,
    domain_labels: torch.Tensor,
    penalty_weight: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """
    Deep CORAL: CE + lambda * sum_{d1,d2} || Cov(feat_d1) - Cov(feat_d2) ||_F^2
    """
    ce_loss = F.cross_entropy(logits, labels)
    domains = domain_labels.unique()
    penalty = torch.tensor(0.0, device=logits.device)

    domain_feats = {}
    for d in domains:
        mask = domain_labels == d
        if mask.sum() > 1:
            domain_feats[d.item()] = features[mask]

    domain_keys = list(domain_feats.keys())
    for i in range(len(domain_keys)):
        for j in range(i + 1, len(domain_keys)):
            f1 = domain_feats[domain_keys[i]]
            f2 = domain_feats[domain_keys[j]]
            # Covariance matrices
            f1_centered = f1 - f1.mean(0, keepdim=True)
            f2_centered = f2 - f2.mean(0, keepdim=True)
            cov1 = (f1_centered.T @ f1_centered) / (f1.shape[0] - 1)
            cov2 = (f2_centered.T @ f2_centered) / (f2.shape[0] - 1)
            penalty = penalty + (cov1 - cov2).pow(2).sum()

    return ce_loss + penalty_weight * penalty / max(1, len(domain_keys) * (len(domain_keys) - 1) / 2)


class DomainDiscriminator(nn.Module):
    """Domain discriminator for DANN."""

    def __init__(self, feat_dim: int, num_domains: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# ============================================================
# 5. Training Loop
# ============================================================


def train_learner(
    model: AdapterWrappedModel,
    train_domains: Dict[str, DataLoader],
    objective: str,
    num_epochs: int = 30,
    lr: float = 5e-5,
    weight_decay: float = 0.0,
    device: torch.device = DEVICE,
) -> Dict:
    """
    Train a single learner with the specified objective.

    Args:
        model: AdapterWrappedModel
        train_domains: dict of domain_name -> DataLoader
        objective: one of 'erm', 'irm', 'coral', 'dann'
        num_epochs: training epochs
        lr: learning rate
        weight_decay: L2 regularization
        device: torch device

    Returns:
        training_stats dict
    """
    model = model.to(device)
    model.train()

    # Optimizer over adapter + head params only
    optimizer = torch.optim.Adam(model.get_trainable_params(), lr=lr, weight_decay=weight_decay)

    # DANN-specific components
    disc = None
    disc_optimizer = None
    grad_rev = None
    if objective == "dann":
        disc = DomainDiscriminator(model.feat_dim, len(train_domains)).to(device)
        disc_optimizer = torch.optim.Adam(disc.parameters(), lr=1e-4)
        grad_rev = GradientReversal(lambda_=1.0)

    domain_names = sorted(train_domains.keys())
    domain_to_idx = {d: i for i, d in enumerate(domain_names)}

    stats = {"train_losses": [], "train_accs": []}

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        # Create iterators for all domains
        domain_iters = {d: iter(dl) for d, dl in train_domains.items()}

        # Train for one epoch (iterate until shortest domain exhausted)
        while True:
            all_x, all_y, all_d = [], [], []
            exhausted = False
            for d_name in domain_names:
                try:
                    x, y = next(domain_iters[d_name])
                except StopIteration:
                    exhausted = True
                    break
                all_x.append(x)
                all_y.append(y)
                all_d.append(torch.full((x.size(0),), domain_to_idx[d_name], dtype=torch.long))

            if exhausted:
                break

            x = torch.cat(all_x, dim=0).to(device)
            y = torch.cat(all_y, dim=0).to(device)
            d = torch.cat(all_d, dim=0).to(device)

            logits = model(x)

            if objective == "erm":
                loss = erm_loss(logits, y)

            elif objective == "irm":
                loss = irm_loss(logits, y, domain_labels=d, penalty_weight=1.0)

            elif objective == "coral":
                # Extract features from penultimate layer
                with torch.no_grad():
                    # Re-forward to get features (use last intermediate)
                    last_idx = max(model._intermediates.keys())
                    feats = model._intermediates[last_idx]
                    if feats.dim() == 4:  # CNN: (B, C, H, W) -> (B, C)
                        feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
                    elif feats.dim() == 3:  # Transformer: (B, T, D) -> (B, D)
                        feats = feats[:, 0]  # CLS token or mean

                # Detach and re-enable grad for CORAL penalty
                feats_for_coral = feats.detach().requires_grad_(True)
                loss = coral_loss(logits, y, feats_for_coral, d, penalty_weight=1.0)

            elif objective == "dann":
                ce_loss = F.cross_entropy(logits, y)

                # Domain discrimination on reversed features
                last_idx = max(model._intermediates.keys())
                feats = model._intermediates[last_idx]
                if feats.dim() == 4:
                    feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
                elif feats.dim() == 3:
                    feats = feats[:, 0]

                rev_feats = grad_rev(feats)
                domain_logits = disc(rev_feats)
                disc_loss = F.cross_entropy(domain_logits, d)

                loss = ce_loss + disc_loss

                disc_optimizer.zero_grad()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if objective == "dann" and disc_optimizer is not None:
                disc_optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            epoch_correct += (preds == y).sum().item()
            epoch_total += x.size(0)

        avg_loss = epoch_loss / max(epoch_total, 1)
        avg_acc = epoch_correct / max(epoch_total, 1)
        stats["train_losses"].append(avg_loss)
        stats["train_accs"].append(avg_acc)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")

    return stats


# ============================================================
# 6. Evaluation
# ============================================================


@torch.no_grad()
def evaluate_domain(model: nn.Module, dataloader: DataLoader, device=DEVICE) -> float:
    """Evaluate accuracy on a single domain."""
    model.eval()
    correct = 0
    total = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_ood(
    model: nn.Module,
    test_domains: Dict[str, DataLoader],
    device=DEVICE,
) -> Dict[str, float]:
    """Evaluate OOD accuracy on each test domain."""
    results = {}
    for d_name, dl in test_domains.items():
        results[d_name] = evaluate_domain(model, dl, device)
    return results


# ============================================================
# 7. Circuit Extraction Wrapper
# ============================================================


@torch.no_grad()
def extract_circuits_ace(
    model: AdapterWrappedModel,
    domain_data: DataLoader,
    num_classes: int = NUM_CLASSES,
    top_k: int = 5,
    edge_threshold: float = 1e-4,
    num_samples_per_class: int = 64,
    device=DEVICE,
) -> List[Dict]:
    """
    Extract class-specific circuits using ACE (activation patching).

    This is a self-contained implementation of the ACE procedure:
      Stage 1: Node importance scoring via zero-ablation
      Stage 2: Inter-layer edge estimation

    Args:
        model: trained AdapterWrappedModel
        domain_data: DataLoader for a specific domain
        num_classes: number of classes
        top_k: top-K units to retain per layer
        edge_threshold: minimum |w| for edge inclusion
        num_samples_per_class: samples per class for patching
        device: torch device

    Returns:
        List of c circuit dicts, each with 'nodes', 'edges', 'importance'
    """
    model.eval()
    model = model.to(device)

    # Collect per-class data
    class_data = defaultdict(list)
    for x, y in domain_data:
        for i in range(x.size(0)):
            label = y[i].item()
            if len(class_data[label]) < num_samples_per_class:
                class_data[label].append(x[i])
    for k in class_data:
        class_data[k] = torch.stack(class_data[k]).to(device)

    adapter_indices = model.adapter_indices
    circuits = []

    for cls in range(num_classes):
        if cls not in class_data or len(class_data[cls]) == 0:
            circuits.append({"nodes": [], "edges": [], "importance": {}})
            continue

        x_cls = class_data[cls]

        # --- Stage 1: Node importance scoring ---
        # Get clean logits
        clean_logits = model(x_cls)
        clean_score = clean_logits[:, cls].mean().item()

        top_units = {}  # layer_idx -> list of (unit_type, unit_id, importance)

        for layer_idx in adapter_indices:
            adapter = model.adapters[str(layer_idx)]
            importances = []

            # Get adapter hidden dim
            hidden_dim = adapter.down.out_features

            # Ablate each adapter neuron
            for neuron_id in range(hidden_dim):
                # Hook to zero out neuron
                original_weight = adapter.up.weight.data[:, neuron_id].clone()
                adapter.up.weight.data[:, neuron_id] = 0.0

                ablated_logits = model(x_cls)
                ablated_score = ablated_logits[:, cls].mean().item()

                # Restore
                adapter.up.weight.data[:, neuron_id] = original_weight

                delta = clean_score - ablated_score
                importances.append(("mlp", neuron_id, delta))

            # For ViT: also ablate attention heads (if applicable)
            if model._arch_type == "vit":
                # Access attention heads in the corresponding block group
                block_group = model.backbone_blocks[layer_idx]
                for block in block_group:
                    if hasattr(block, "attn"):
                        num_heads = block.attn.num_heads
                        head_dim = block.attn.head_dim
                        for head_id in range(num_heads):
                            # Zero out head's QKV projections temporarily
                            qkv = block.attn.qkv
                            orig_weight = qkv.weight.data.clone()
                            orig_bias = qkv.bias.data.clone() if qkv.bias is not None else None

                            # Zero the portion corresponding to this head
                            dim = qkv.weight.shape[0] // 3
                            for offset in [0, dim, 2 * dim]:  # Q, K, V
                                start = offset + head_id * head_dim
                                end = start + head_dim
                                qkv.weight.data[start:end] = 0.0
                                if orig_bias is not None:
                                    qkv.bias.data[start:end] = 0.0

                            ablated_logits = model(x_cls)
                            ablated_score = ablated_logits[:, cls].mean().item()

                            # Restore
                            qkv.weight.data = orig_weight
                            if orig_bias is not None:
                                qkv.bias.data = orig_bias

                            delta = clean_score - ablated_score
                            importances.append(("attn", head_id, delta))

            # Select top-K by absolute importance
            importances.sort(key=lambda t: abs(t[2]), reverse=True)
            top_units[layer_idx] = importances[:top_k]

        # --- Stage 2: Edge estimation ---
        edges = []
        sorted_layers = sorted(top_units.keys())

        for li in range(len(sorted_layers) - 1):
            l1 = sorted_layers[li]
            l2 = sorted_layers[li + 1]

            for u1_type, u1_id, u1_imp in top_units[l1]:
                # Ablate u1
                if u1_type == "mlp":
                    adapter1 = model.adapters[str(l1)]
                    orig = adapter1.up.weight.data[:, u1_id].clone()
                    adapter1.up.weight.data[:, u1_id] = 0.0
                else:
                    continue  # Simplified: skip attn-to-attn for edge estimation

                # Measure effect on u2 activations
                # Run forward and capture adapter activations at l2
                _ = model(x_cls)

                for u2_type, u2_id, u2_imp in top_units[l2]:
                    if u2_type != u1_type:
                        continue  # Same-type edges only

                    # Compute edge weight as change in u2's mean activation
                    # (Simplified: use importance product as proxy)
                    edge_weight = u1_imp * u2_imp
                    if abs(edge_weight) > edge_threshold:
                        edges.append({
                            "source": (l1, u1_type, u1_id),
                            "target": (l2, u2_type, u2_id),
                            "weight": edge_weight,
                        })

                # Restore u1
                if u1_type == "mlp":
                    adapter1.up.weight.data[:, u1_id] = orig

        # Build circuit
        nodes = []
        importance_dict = {}
        for layer_idx, units in top_units.items():
            for u_type, u_id, imp in units:
                node = (layer_idx, u_type, u_id)
                nodes.append(node)
                importance_dict[node] = imp

        circuits.append({
            "nodes": nodes,
            "edges": edges,
            "importance": importance_dict,
        })

    return circuits


# ============================================================
# 8. Baseline Metrics
# ============================================================


@torch.no_grad()
def extract_features(
    model: nn.Module, dataloader: DataLoader, device=DEVICE
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract penultimate-layer features and labels."""
    model.eval()
    all_feats, all_labels = [], []
    for x, y in dataloader:
        x = x.to(device)
        _ = model(x)
        last_idx = max(model._intermediates.keys())
        feats = model._intermediates[last_idx]
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        elif feats.dim() == 3:
            feats = feats[:, 0]
        all_feats.append(feats.cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between two feature matrices.
    CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K,K) * HSIC(L,L))
    where K = X @ X.T, L = Y @ Y.T
    """
    n = X.shape[0]
    assert Y.shape[0] == n

    # Center
    H = np.eye(n) - np.ones((n, n)) / n
    K = X @ X.T
    L = Y @ Y.T

    KH = K @ H
    LH = L @ H

    hsic_kl = np.trace(KH @ LH) / (n - 1) ** 2
    hsic_kk = np.trace(KH @ KH) / (n - 1) ** 2
    hsic_ll = np.trace(LH @ LH) / (n - 1) ** 2

    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-10:
        return 0.0
    return float(hsic_kl / denom)


def compute_svcca(X: np.ndarray, Y: np.ndarray, threshold: float = 0.99) -> float:
    """
    SVCCA: SVD on each matrix, then CCA on truncated components.
    Returns mean canonical correlation.
    """
    from numpy.linalg import svd

    def truncated_svd(M, threshold):
        U, s, Vt = svd(M, full_matrices=False)
        total = s.sum()
        cumsum = np.cumsum(s)
        k = np.searchsorted(cumsum, threshold * total) + 1
        return U[:, :k] @ np.diag(s[:k])

    X_red = truncated_svd(X, threshold)
    Y_red = truncated_svd(Y, threshold)

    k = min(X_red.shape[1], Y_red.shape[1])
    X_red = X_red[:, :k]
    Y_red = Y_red[:, :k]

    # CCA via QR + SVD
    Qx, _ = np.linalg.qr(X_red)
    Qy, _ = np.linalg.qr(Y_red)

    _, s, _ = svd(Qx.T @ Qy)
    return float(s.mean())


def compute_rsa(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Representational Similarity Analysis: Spearman correlation between
    pairwise distance matrices.
    """
    from scipy.spatial.distance import pdist

    dx = pdist(X, metric="correlation")
    dy = pdist(Y, metric="correlation")
    rho, _ = spearmanr(dx, dy)
    return float(rho) if not np.isnan(rho) else 0.0


def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """
    Maximum Mean Discrepancy with Gaussian kernel.
    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    Returns negative MMD (so higher = more aligned = better).
    """
    from scipy.spatial.distance import cdist

    n, m = X.shape[0], Y.shape[0]

    Kxx = np.exp(-gamma * cdist(X, X, "sqeuclidean"))
    Kyy = np.exp(-gamma * cdist(Y, Y, "sqeuclidean"))
    Kxy = np.exp(-gamma * cdist(X, Y, "sqeuclidean"))

    mmd2 = Kxx.sum() / (n * n) + Kyy.sum() / (m * m) - 2 * Kxy.sum() / (n * m)
    return float(-mmd2)  # Negate so higher = better aligned


def compute_gradient_norm(model: nn.Module, dataloader: DataLoader, device=DEVICE) -> float:
    """Average gradient norm of trainable parameters."""
    model.train()
    total_norm = 0.0
    count = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        norm = 0.0
        for p in model.get_trainable_params():
            if p.grad is not None:
                norm += p.grad.data.norm(2).item() ** 2
        total_norm += norm ** 0.5
        count += 1
        model.zero_grad()
    model.eval()
    return total_norm / max(count, 1)


def compute_fisher_flatness(
    model: nn.Module, dataloader: DataLoader, device=DEVICE
) -> float:
    """
    Fisher information trace as a flatness measure.
    Lower trace = flatter = better expected generalization.
    Returns negative trace (so higher = flatter = better).
    """
    model.eval()
    fisher_trace = 0.0
    count = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        model.zero_grad()
        loss.backward()
        for p in model.get_trainable_params():
            if p.grad is not None:
                fisher_trace += (p.grad.data ** 2).sum().item()
        count += x.size(0)
    return float(-fisher_trace / max(count, 1))


def compute_cross_domain_baseline(
    model: AdapterWrappedModel,
    train_domain_loaders: Dict[str, DataLoader],
    metric_fn,
    device=DEVICE,
) -> float:
    """
    Compute a cross-domain representational similarity metric.
    Averages over all domain pairs.
    """
    domains = sorted(train_domain_loaders.keys())
    K = len(domains)

    # Extract features for each domain
    domain_feats = {}
    for d_name in domains:
        feats, _ = extract_features(model, train_domain_loaders[d_name], device)
        domain_feats[d_name] = feats

    # Average pairwise metric
    values = []
    for i in range(K):
        for j in range(i + 1, K):
            # Subsample to same size
            n = min(domain_feats[domains[i]].shape[0], domain_feats[domains[j]].shape[0])
            fi = domain_feats[domains[i]][:n]
            fj = domain_feats[domains[j]][:n]
            values.append(metric_fn(fi, fj))

    return float(np.mean(values))


# ============================================================
# 9. Main Experiment Pipeline
# ============================================================


def build_learner_pool(
    data_root: str,
    target_domain: str,
    num_epochs: int = 30,
    batch_size: int = 32,
    lr: float = 5e-5,
    device=DEVICE,
) -> List[Dict]:
    """
    Build and train the full pool of 48 learners for a given target domain.

    Returns list of dicts with keys:
        'model', 'arch', 'objective', 'regularization',
        'train_stats', 'ood_acc', 'train_acc'
    """
    source_domains = [d for d in PACS_DOMAINS if d != target_domain]
    print(f"\n{'='*60}")
    print(f"Target domain: {target_domain}")
    print(f"Source domains: {source_domains}")
    print(f"{'='*60}")

    # Load data
    train_loaders = {}
    eval_loaders = {}
    for d in source_domains:
        train_ds = load_pacs_domain(data_root, d, train=True)
        eval_ds = load_pacs_domain(data_root, d, train=False)
        train_loaders[d] = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        eval_loaders[d] = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    test_ds = load_pacs_domain(data_root, target_domain, train=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    learner_pool = []

    for arch in ARCHITECTURES:
        for obj in OBJECTIVES:
            for reg in REGULARIZATIONS:
                print(f"\n--- Training: {arch} / {obj} / {reg} ---")

                dropout_rate = 0.5 if reg == "dropout" else 0.0
                wd = 1e-4 if reg == "weight_decay" else 0.0

                try:
                    model = AdapterWrappedModel(
                        arch=arch,
                        num_classes=NUM_CLASSES,
                        bottleneck_ratio=4,
                        num_adapter_layers=4,
                        dropout_rate=dropout_rate,
                    )
                except (ImportError, Exception) as e:
                    print(f"  Skipping {arch}: {e}")
                    continue

                train_stats = train_learner(
                    model=model,
                    train_domains=train_loaders,
                    objective=obj,
                    num_epochs=num_epochs,
                    lr=lr,
                    weight_decay=wd,
                    device=device,
                )

                # Evaluate
                ood_acc = evaluate_domain(model, test_loader, device)
                train_accs = {}
                for d in source_domains:
                    train_accs[d] = evaluate_domain(model, eval_loaders[d], device)
                avg_train_acc = np.mean(list(train_accs.values()))

                print(f"  OOD acc ({target_domain}): {ood_acc:.4f}")
                print(f"  Avg train acc: {avg_train_acc:.4f}")

                learner_pool.append({
                    "model": model,
                    "arch": arch,
                    "objective": obj,
                    "regularization": reg,
                    "train_stats": train_stats,
                    "ood_acc": ood_acc,
                    "train_acc": avg_train_acc,
                    "train_accs_per_domain": train_accs,
                })

    return learner_pool


def compute_all_metrics(
    learner_pool: List[Dict],
    train_domain_loaders: Dict[str, DataLoader],
    graph_kernel_fn=None,
    top_k: int = 5,
    edge_threshold: float = 1e-4,
    cas_h_alpha: float = 10.0,
    device=DEVICE,
) -> List[Dict]:
    """
    Compute CAS, CAS-H, and all baseline metrics for every learner.

    Args:
        learner_pool: list of learner dicts from build_learner_pool
        train_domain_loaders: dict of domain -> DataLoader (eval mode)
        graph_kernel_fn: callable(circuit1, circuit2) -> float
            If None, uses a simple Jaccard-based proxy kernel.
        top_k: top-K for circuit extraction
        edge_threshold: edge threshold for circuit extraction
        cas_h_alpha: softmax temperature for CAS-H
        device: torch device

    Returns:
        List of dicts with all metric values added
    """
    domains = sorted(train_domain_loaders.keys())
    K = len(domains)

    # Default graph kernel: Jaccard over node sets + weighted edge overlap
    if graph_kernel_fn is None:
        def graph_kernel_fn(c1, c2):
            """Simple graph kernel: weighted Jaccard of node/edge sets."""
            nodes1 = set(c1["nodes"])
            nodes2 = set(c2["nodes"])
            if len(nodes1) == 0 and len(nodes2) == 0:
                return 1.0
            node_jaccard = len(nodes1 & nodes2) / max(len(nodes1 | nodes2), 1)

            # Edge overlap
            edges1 = {(e["source"], e["target"]): e["weight"] for e in c1["edges"]}
            edges2 = {(e["source"], e["target"]): e["weight"] for e in c2["edges"]}
            common_edges = set(edges1.keys()) & set(edges2.keys())
            if len(common_edges) == 0:
                edge_sim = 0.0
            else:
                weight_sim = sum(
                    1.0 - abs(edges1[e] - edges2[e]) / (abs(edges1[e]) + abs(edges2[e]) + 1e-10)
                    for e in common_edges
                ) / max(len(set(edges1.keys()) | set(edges2.keys())), 1)
                edge_sim = weight_sim

            return 0.5 * node_jaccard + 0.5 * edge_sim

    for li, learner in enumerate(learner_pool):
        model = learner["model"]
        print(f"\nComputing metrics for learner {li+1}/{len(learner_pool)}: "
              f"{learner['arch']}/{learner['objective']}/{learner['regularization']}")

        # --- Circuit extraction & CAS ---
        circuit_families = {}
        for d_name in domains:
            circuits = extract_circuits_ace(
                model, train_domain_loaders[d_name],
                num_classes=NUM_CLASSES, top_k=top_k,
                edge_threshold=edge_threshold, device=device,
            )
            circuit_families[d_name] = circuits

        # Compute cross-domain CAS and CAS-H
        cas_values = []
        cas_h_values = []
        all_S_matrices = []

        for i in range(K):
            for j in range(i + 1, K):
                d_i, d_j = domains[i], domains[j]
                c = NUM_CLASSES
                S = np.zeros((c, c))
                for ci in range(c):
                    for cj in range(c):
                        S[ci, cj] = graph_kernel_fn(
                            circuit_families[d_i][ci],
                            circuit_families[d_j][cj],
                        )
                cas_values.append(compute_cas(S))
                cas_h_values.append(compute_cas_h(S, alpha=cas_h_alpha))
                all_S_matrices.append({"pair": (d_i, d_j), "S": S})

        learner["cross_cas"] = float(np.mean(cas_values))
        learner["cross_cas_h"] = float(np.mean(cas_h_values))
        learner["cas_for_ranking"] = learner["cross_cas"]
        learner["cas_h_for_ranking"] = 1.0 - learner["cross_cas_h"]  # Invert for ranking
        learner["S_matrices"] = all_S_matrices

        # Per-class diagonal coherence (from first S matrix)
        if all_S_matrices:
            learner["class_diagonal_coherence"] = {
                PACS_CLASSES[i]: float(all_S_matrices[0]["S"][i, i])
                for i in range(NUM_CLASSES)
            }
            learner["class_entropies"] = compute_row_entropies(
                all_S_matrices[0]["S"], alpha=cas_h_alpha
            ).tolist()

        # --- Baseline: CKA ---
        learner["cka"] = compute_cross_domain_baseline(
            model, train_domain_loaders, compute_cka, device
        )

        # --- Baseline: SVCCA ---
        learner["svcca"] = compute_cross_domain_baseline(
            model, train_domain_loaders, compute_svcca, device
        )

        # --- Baseline: RSA ---
        learner["rsa"] = compute_cross_domain_baseline(
            model, train_domain_loaders, compute_rsa, device
        )

        # --- Baseline: MMD alignment ---
        learner["mmd"] = compute_cross_domain_baseline(
            model, train_domain_loaders, compute_mmd, device
        )

        # --- Baseline: Gradient norm ---
        combined_loader = DataLoader(
            ConcatDataset([dl.dataset for dl in train_domain_loaders.values()]),
            batch_size=32, shuffle=False, num_workers=2,
        )
        learner["grad_norm"] = -compute_gradient_norm(model, combined_loader, device)

        # --- Baseline: Fisher flatness ---
        learner["fisher_flatness"] = compute_fisher_flatness(model, combined_loader, device)

        # --- Baseline: Parameter count ---
        learner["num_params"] = sum(
            p.numel() for p in model.get_trainable_params()
        )

        print(f"  CAS={learner['cross_cas']:.4f}, CAS-H={learner['cross_cas_h']:.4f}, "
              f"CKA={learner['cka']:.4f}, OOD={learner['ood_acc']:.4f}")

    return learner_pool


# ============================================================
# 10. Rank Correlation & Statistical Tests
# ============================================================


def compute_spearman_with_ci(
    metric_values: np.ndarray,
    ood_accs: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> Dict:
    """
    Compute Spearman rank correlation with bootstrap confidence interval.

    Returns dict with 'rho', 'pvalue', 'ci_low', 'ci_high'.
    """
    rho, pvalue = spearmanr(metric_values, ood_accs)

    # Bootstrap CI
    n = len(metric_values)
    rng = np.random.default_rng(42)
    bootstrap_rhos = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = spearmanr(metric_values[idx], ood_accs[idx])
        if not np.isnan(r):
            bootstrap_rhos.append(r)

    alpha = 1 - ci_level
    ci_low = np.percentile(bootstrap_rhos, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_rhos, 100 * (1 - alpha / 2))

    return {
        "rho": float(rho),
        "pvalue": float(pvalue),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def permutation_test(
    metric_values: np.ndarray,
    ood_accs: np.ndarray,
    n_permutations: int = 10000,
) -> float:
    """
    Permutation test p-value for Spearman correlation.
    """
    observed_rho, _ = spearmanr(metric_values, ood_accs)
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_permutations):
        perm_accs = rng.permutation(ood_accs)
        perm_rho, _ = spearmanr(metric_values, perm_accs)
        if abs(perm_rho) >= abs(observed_rho):
            count += 1
    return count / n_permutations


def evaluate_all_correlations(
    learner_pool: List[Dict], target_domain: str
) -> Dict:
    """
    Compute Spearman correlations for all metrics against OOD accuracy.
    """
    ood_accs = np.array([l["ood_acc"] for l in learner_pool])

    metrics = {
        "CAS (ours)": np.array([l["cas_for_ranking"] for l in learner_pool]),
        "1 - CAS-H (ours)": np.array([l["cas_h_for_ranking"] for l in learner_pool]),
        "CKA": np.array([l["cka"] for l in learner_pool]),
        "SVCCA": np.array([l["svcca"] for l in learner_pool]),
        "RSA": np.array([l["rsa"] for l in learner_pool]),
        "MMD alignment": np.array([l["mmd"] for l in learner_pool]),
        "Training accuracy": np.array([l["train_acc"] for l in learner_pool]),
        "# Parameters": np.array([l["num_params"] for l in learner_pool]),
        "Gradient norm": np.array([l["grad_norm"] for l in learner_pool]),
        "Fisher flatness": np.array([l["fisher_flatness"] for l in learner_pool]),
    }

    results = {}
    print(f"\n{'='*60}")
    print(f"Spearman Correlations vs OOD Accuracy (Target: {target_domain})")
    print(f"{'='*60}")
    print(f"{'Metric':<25s} {'rho':>8s} {'p-value':>10s} {'95% CI':>20s}")
    print(f"{'-'*65}")

    for name, values in metrics.items():
        corr = compute_spearman_with_ci(values, ood_accs)
        perm_p = permutation_test(values, ood_accs)
        corr["perm_pvalue"] = perm_p
        results[name] = corr

        ci_str = f"[{corr['ci_low']:.3f}, {corr['ci_high']:.3f}]"
        print(f"{name:<25s} {corr['rho']:>8.4f} {perm_p:>10.4f} {ci_str:>20s}")

    return results


# ============================================================
# 11. Ablation Studies
# ============================================================


def ablation_top_k(
    learner_pool: List[Dict],
    train_domain_loaders: Dict[str, DataLoader],
    graph_kernel_fn=None,
    k_values: List[int] = [3, 5, 10, 15, 20],
    device=DEVICE,
) -> Dict[int, float]:
    """Ablation: effect of top-K on mean Spearman correlation."""
    results = {}
    ood_accs = np.array([l["ood_acc"] for l in learner_pool])
    domains = sorted(train_domain_loaders.keys())

    for K in k_values:
        print(f"\n  Ablation: top_k = {K}")
        cas_scores = []
        for learner in learner_pool:
            model = learner["model"]
            circuit_families = {}
            for d in domains:
                circuits = extract_circuits_ace(
                    model, train_domain_loaders[d],
                    top_k=K, device=device,
                )
                circuit_families[d] = circuits
            cross_cas, _ = compute_cross_domain_cas(
                circuit_families, graph_kernel_fn or _default_kernel, domains,
            )
            cas_scores.append(cross_cas)

        rho, _ = spearmanr(cas_scores, ood_accs)
        results[K] = float(rho)
        print(f"    Spearman rho = {rho:.4f}")

    return results


def ablation_edge_threshold(
    learner_pool: List[Dict],
    train_domain_loaders: Dict[str, DataLoader],
    graph_kernel_fn=None,
    eps_values: List[float] = [1e-3, 1e-4, 1e-5],
    device=DEVICE,
) -> Dict[float, float]:
    """Ablation: effect of edge threshold on mean Spearman correlation."""
    results = {}
    ood_accs = np.array([l["ood_acc"] for l in learner_pool])
    domains = sorted(train_domain_loaders.keys())

    for eps in eps_values:
        print(f"\n  Ablation: epsilon = {eps}")
        cas_scores = []
        for learner in learner_pool:
            model = learner["model"]
            circuit_families = {}
            for d in domains:
                circuits = extract_circuits_ace(
                    model, train_domain_loaders[d],
                    edge_threshold=eps, device=device,
                )
                circuit_families[d] = circuits
            cross_cas, _ = compute_cross_domain_cas(
                circuit_families, graph_kernel_fn or _default_kernel, domains,
            )
            cas_scores.append(cross_cas)

        rho, _ = spearmanr(cas_scores, ood_accs)
        results[eps] = float(rho)
        print(f"    Spearman rho = {rho:.4f}")

    return results


def ablation_cas_h_temperature(
    learner_pool: List[Dict],
    alpha_values: List[float] = [1, 5, 10, 20, 50, 100],
) -> Dict[float, float]:
    """Ablation: effect of CAS-H temperature on Spearman correlation."""
    ood_accs = np.array([l["ood_acc"] for l in learner_pool])
    results = {}

    for alpha in alpha_values:
        cas_h_scores = []
        for learner in learner_pool:
            S_matrices = learner.get("S_matrices", [])
            if not S_matrices:
                cas_h_scores.append(0.5)
                continue
            cas_h_vals = [compute_cas_h(sm["S"], alpha=alpha) for sm in S_matrices]
            cas_h_scores.append(1.0 - np.mean(cas_h_vals))

        rho, _ = spearmanr(cas_h_scores, ood_accs)
        results[alpha] = float(rho)
        print(f"  CAS-H alpha={alpha}: rho={rho:.4f}")

    return results


# ============================================================
# 12. Visualization
# ============================================================


def plot_scatter_cas_vs_ood(
    learner_pool: List[Dict],
    target_domain: str,
    output_path: str,
):
    """
    Create scatter plot: CAS vs OOD accuracy, colored by training objective.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Colors and markers per objective
    style_map = {
        "erm":   {"color": "#4472C4", "marker": "o", "label": "ERM"},
        "irm":   {"color": "#C0504D", "marker": "^", "label": "IRM"},
        "coral": {"color": "#2E7D32", "marker": "s", "label": "CORAL"},
        "dann":  {"color": "#E68A00", "marker": "D", "label": "DANN"},
    }

    # Left panel: CAS vs OOD
    ax = axes[0]
    for obj in OBJECTIVES:
        subset = [l for l in learner_pool if l["objective"] == obj]
        if not subset:
            continue
        x = [l["cas_for_ranking"] for l in subset]
        y = [l["ood_acc"] * 100 for l in subset]
        ax.scatter(x, y, c=style_map[obj]["color"], marker=style_map[obj]["marker"],
                   s=50, alpha=0.8, label=style_map[obj]["label"], edgecolors="white", linewidth=0.5)

    # Trend line
    all_cas = np.array([l["cas_for_ranking"] for l in learner_pool])
    all_ood = np.array([l["ood_acc"] * 100 for l in learner_pool])
    z = np.polyfit(all_cas, all_ood, 1)
    x_line = np.linspace(all_cas.min(), all_cas.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.5, linewidth=1.5)

    rho, _ = spearmanr(all_cas, all_ood)
    ax.set_xlabel(r"$\overline{\mathrm{CAS}}$ (RWK)", fontsize=12)
    ax.set_ylabel("OOD Accuracy (%)", fontsize=12)
    ax.set_title(f"CAS vs OOD Accuracy (Target: {target_domain.title()})\n"
                 f"$\\rho_S = {rho:.2f}$", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right panel: CKA vs OOD
    ax = axes[1]
    all_cka = np.array([l["cka"] for l in learner_pool])
    ax.scatter(all_cka, all_ood, c="gray", marker="o", s=40, alpha=0.6, edgecolors="white", linewidth=0.5)

    rho_cka, _ = spearmanr(all_cka, all_ood)
    ax.set_xlabel("CKA (cross-domain)", fontsize=12)
    ax.set_ylabel("OOD Accuracy (%)", fontsize=12)
    ax.set_title(f"CKA vs OOD Accuracy (Target: {target_domain.title()})\n"
                 f"$\\rho_S = {rho_cka:.2f}$", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved scatter plot: {output_path}")


def plot_class_vulnerability(
    learner_pool: List[Dict],
    target_domain: str,
    output_path: str,
):
    """Plot per-class diagonal coherence heatmap for selected learners."""
    # Select representative learners: best and worst per objective
    selected = []
    for obj in OBJECTIVES:
        subset = [l for l in learner_pool if l["objective"] == obj]
        if subset:
            subset.sort(key=lambda l: l["ood_acc"])
            selected.append(subset[-1])  # best

    if not selected:
        return

    fig, ax = plt.subplots(figsize=(10, len(selected) * 0.7 + 1.5))

    data = []
    labels = []
    for learner in selected:
        diag = learner.get("class_diagonal_coherence", {})
        row = [diag.get(cls, 0.0) for cls in PACS_CLASSES]
        data.append(row)
        labels.append(f"{learner['arch']}/{learner['objective']}")

    data = np.array(data)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0.3, vmax=1.0)

    ax.set_xticks(range(len(PACS_CLASSES)))
    ax.set_xticklabels(PACS_CLASSES, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(PACS_CLASSES)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if data[i, j] > 0.6 else "white")

    plt.colorbar(im, ax=ax, label="Diagonal Coherence $S_{ii}$")
    ax.set_title(f"Class Vulnerability Profile (Target: {target_domain.title()})", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved vulnerability heatmap: {output_path}")


def plot_ablation_summary(ablation_results: Dict, output_path: str):
    """Plot ablation results as a grouped bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Top-K ablation
    if "top_k" in ablation_results:
        ax = axes[0]
        k_vals = sorted(ablation_results["top_k"].keys())
        rhos = [ablation_results["top_k"][k] for k in k_vals]
        ax.bar(range(len(k_vals)), rhos, color="#4472C4", alpha=0.8)
        ax.set_xticks(range(len(k_vals)))
        ax.set_xticklabels([str(k) for k in k_vals])
        ax.set_xlabel("Top-K units per layer")
        ax.set_ylabel(r"Spearman $\rho_S$")
        ax.set_title("Effect of Top-K")
        ax.set_ylim(0.7, 1.0)

    # Edge threshold ablation
    if "edge_threshold" in ablation_results:
        ax = axes[1]
        eps_vals = sorted(ablation_results["edge_threshold"].keys())
        rhos = [ablation_results["edge_threshold"][e] for e in eps_vals]
        ax.bar(range(len(eps_vals)), rhos, color="#2E7D32", alpha=0.8)
        ax.set_xticks(range(len(eps_vals)))
        ax.set_xticklabels([f"{e:.0e}" for e in eps_vals])
        ax.set_xlabel(r"Edge threshold $\epsilon$")
        ax.set_ylabel(r"Spearman $\rho_S$")
        ax.set_title(r"Effect of $\epsilon$")
        ax.set_ylim(0.7, 1.0)

    # CAS-H temperature ablation
    if "cas_h_temperature" in ablation_results:
        ax = axes[2]
        alpha_vals = sorted(ablation_results["cas_h_temperature"].keys())
        rhos = [ablation_results["cas_h_temperature"][a] for a in alpha_vals]
        ax.bar(range(len(alpha_vals)), rhos, color="#C0504D", alpha=0.8)
        ax.set_xticks(range(len(alpha_vals)))
        ax.set_xticklabels([str(int(a)) for a in alpha_vals])
        ax.set_xlabel(r"CAS-H temperature $\alpha$")
        ax.set_ylabel(r"Spearman $\rho_S$")
        ax.set_title(r"Effect of $\alpha$")
        ax.set_ylim(0.7, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved ablation summary: {output_path}")


def generate_results_table(all_results: Dict, output_path: str):
    """Generate the main results table (Table 2 in the paper) as LaTeX and CSV."""
    metrics = [
        "CAS (ours)", "1 - CAS-H (ours)",
        "CKA", "SVCCA", "RSA",
        "MMD alignment", "Training accuracy", "# Parameters",
        "Gradient norm", "Fisher flatness",
    ]

    targets = sorted(all_results.keys())

    # CSV
    with open(output_path + ".csv", "w") as f:
        f.write("Metric," + ",".join(f"Tgt:{t}" for t in targets) + ",Mean\n")
        for m in metrics:
            vals = []
            for t in targets:
                rho = all_results[t].get(m, {}).get("rho", float("nan"))
                vals.append(rho)
            mean_rho = np.nanmean(vals)
            f.write(f"{m}," + ",".join(f"{v:.4f}" for v in vals) + f",{mean_rho:.4f}\n")

    # LaTeX
    with open(output_path + ".tex", "w") as f:
        f.write("\\begin{tabular}{l" + "c" * len(targets) + "|c}\n\\toprule\n")
        f.write("\\textbf{Metric} & " +
                " & ".join(f"\\textbf{{Tgt: {t[0].upper()}}}" for t in targets) +
                " & \\textbf{Mean} \\\\\n\\midrule\n")
        for m in metrics:
            vals = []
            for t in targets:
                rho = all_results[t].get(m, {}).get("rho", float("nan"))
                vals.append(rho)
            mean_rho = np.nanmean(vals)
            vals_str = " & ".join(f"{v:.2f}" for v in vals)
            f.write(f"{m} & {vals_str} & {mean_rho:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"  Saved results table: {output_path}.csv and {output_path}.tex")


# ============================================================
# 13. Main Entry Point
# ============================================================


def _default_kernel(c1, c2):
    """Fallback kernel for ablations."""
    nodes1 = set(c1.get("nodes", []))
    nodes2 = set(c2.get("nodes", []))
    if len(nodes1) == 0 and len(nodes2) == 0:
        return 1.0
    return len(nodes1 & nodes2) / max(len(nodes1 | nodes2), 1)


def main():
    parser = argparse.ArgumentParser(
        description="Circuit Alignment Predicts OOD Generalization - Experiments"
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing PACS dataset")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory for saving results")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Training epochs per learner")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--edge_threshold", type=float, default=1e-4)
    parser.add_argument("--cas_h_alpha", type=float, default=10.0)
    parser.add_argument("--target_domains", type=str, nargs="+",
                        default=PACS_DOMAINS,
                        help="Target domains to evaluate (default: all)")
    parser.add_argument("--skip_ablations", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results = {}
    all_learner_pools = {}

    # ---- Main experiment: one run per target domain ----
    for target in args.target_domains:
        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT: Target = {target}")
        print(f"{'#'*70}")

        source_domains = [d for d in PACS_DOMAINS if d != target]

        # 1. Build & train learner pool
        learner_pool = build_learner_pool(
            data_root=args.data_root,
            target_domain=target,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )

        if len(learner_pool) == 0:
            print(f"  WARNING: No learners trained for target={target}. Skipping.")
            continue

        # 2. Compute all metrics
        eval_loaders = {}
        for d in source_domains:
            eval_ds = load_pacs_domain(args.data_root, d, train=False)
            eval_loaders[d] = DataLoader(eval_ds, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)

        learner_pool = compute_all_metrics(
            learner_pool=learner_pool,
            train_domain_loaders=eval_loaders,
            top_k=args.top_k,
            edge_threshold=args.edge_threshold,
            cas_h_alpha=args.cas_h_alpha,
            device=device,
        )

        # 3. Evaluate correlations
        results = evaluate_all_correlations(learner_pool, target)
        all_results[target] = results

        # 4. Visualizations
        plot_scatter_cas_vs_ood(
            learner_pool, target,
            os.path.join(args.output_dir, f"scatter_{target}.png"),
        )
        plot_class_vulnerability(
            learner_pool, target,
            os.path.join(args.output_dir, f"vulnerability_{target}.png"),
        )

        all_learner_pools[target] = learner_pool

        # 5. Save per-target results
        serializable = []
        for l in learner_pool:
            sl = {k: v for k, v in l.items()
                  if k not in ("model", "S_matrices", "train_stats")}
            serializable.append(sl)
        with open(os.path.join(args.output_dir, f"learners_{target}.json"), "w") as f:
            json.dump(serializable, f, indent=2, default=str)

    # ---- Ablation studies ----
    if not args.skip_ablations and all_learner_pools:
        print(f"\n{'#'*70}")
        print(f"# ABLATION STUDIES")
        print(f"{'#'*70}")

        # Use first target's pool for ablations
        target_for_ablation = list(all_learner_pools.keys())[0]
        pool = all_learner_pools[target_for_ablation]
        source_domains = [d for d in PACS_DOMAINS if d != target_for_ablation]
        abl_loaders = {}
        for d in source_domains:
            ds = load_pacs_domain(args.data_root, d, train=False)
            abl_loaders[d] = DataLoader(ds, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)

        ablation_results = {}

        # CAS-H temperature (cheap, no re-extraction needed)
        print("\n--- Ablation: CAS-H temperature ---")
        ablation_results["cas_h_temperature"] = ablation_cas_h_temperature(pool)

        # Top-K (expensive: re-extracts circuits)
        print("\n--- Ablation: top-K ---")
        ablation_results["top_k"] = ablation_top_k(
            pool, abl_loaders, k_values=[3, 5, 10, 15, 20], device=device,
        )

        # Edge threshold
        print("\n--- Ablation: edge threshold ---")
        ablation_results["edge_threshold"] = ablation_edge_threshold(
            pool, abl_loaders, eps_values=[1e-3, 1e-4, 1e-5], device=device,
        )

        plot_ablation_summary(
            ablation_results,
            os.path.join(args.output_dir, "ablation_summary.png"),
        )

        with open(os.path.join(args.output_dir, "ablation_results.json"), "w") as f:
            json.dump(ablation_results, f, indent=2, default=str)

    # ---- Generate main results table ----
    if all_results:
        generate_results_table(
            all_results,
            os.path.join(args.output_dir, "main_results_table"),
        )

    # ---- Save full summary ----
    summary = {
        "target_domains": list(all_results.keys()),
        "correlations": {
            target: {metric: vals for metric, vals in results.items()}
            for target, results in all_results.items()
        },
    }
    with open(os.path.join(args.output_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"All experiments complete. Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

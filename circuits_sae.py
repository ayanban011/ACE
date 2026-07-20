import os
 
import copy
import itertools
import concurrent.futures
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import networkx as nx
from pyvis.network import Network

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
NUM_CLASSES = 65
EPOCHS = 10000
TOPK = 5   # important neurons per layer
PATIENCE = 30
SAVE_DIR = "./outputs_officehome"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# MULTI-GPU CONFIG
# -----------------------------
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
USE_MULTI_GPU = NUM_GPUS > 1
print(f"Detected {NUM_GPUS} visible GPU(s). Multi-GPU: {USE_MULTI_GPU}")

def get_available_devices():
    """List of torch devices to spread work across."""
    if torch.cuda.is_available():
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return [torch.device("cpu")]

def base_module(model):
    """Unwrap nn.DataParallel to get the real module (so `.adapters` etc.
    stay accessible regardless of whether the model is wrapped)."""
    return model.module if isinstance(model, nn.DataParallel) else model

def maybe_data_parallel(model):
    """Wrap the classification model in nn.DataParallel for training/eval
    ONLY. Do not use the wrapped object for hook-based circuit tracing --
    forward hooks on submodules of a DataParallel-wrapped model run on
    per-GPU replicas and are not safe/reliable for the cache-dict pattern
    used by AdapterCircuitTracer / SAECircuitTracer. Always pass
    base_module(model) into the tracers."""
    if torch.cuda.is_available() and len(get_available_devices()) > 1:
        return nn.DataParallel(model)
    return model

# -----------------------------
# DATA
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

dataset = datasets.ImageFolder("./OfficeHomeDataset_10072016/Art", transform=transform)
 
generator = torch.Generator().manual_seed(42)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
 
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers = 0, pin_memory=True)

# -----------------------------
# MODEL
# -----------------------------
class MLPAdapter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 16)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels)
        )
 
    def forward(self, x):
        B,C,H,W = x.shape
        x_flat = x.permute(0,2,3,1).reshape(-1,C)
        out = self.mlp(x_flat)
        out = out.view(B,H,W,C).permute(0,3,1,2)
        return x + out
    

class VGG19_Adapter(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = models.vgg19(weights="IMAGENET1K_V1")
 
        for p in base.features.parameters():
            p.requires_grad = False
 
        self.features = base.features

        # SAFE adapter naming (no dots)
        self.adapter_layers = [28, 30, 32]
        self.adapters = nn.ModuleDict({
            str(i): MLPAdapter(512) for i in self.adapter_layers
        })
 
        self.pool = base.avgpool
 
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 1024), nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
 
    def forward(self, x):
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in self.adapters:
                x = self.adapters[name](x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        return self.classifier(x)
 
model = VGG19_Adapter(NUM_CLASSES).to(DEVICE)
model = maybe_data_parallel(model)  # spreads batches across all visible GPUs for train/eval only

import math
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def _flatten_spatial(o):
    """[B,C,H,W] -> ([B*H*W, C], (B,H,W))"""
    B, C, H, W = o.shape
    return o.permute(0, 2, 3, 1).reshape(-1, C), (B, H, W)

def _unflatten_spatial(flat, shape, C):
    """[B*H*W, C] -> [B,C,H,W]"""
    B, H, W = shape
    return flat.view(B, H, W, C).permute(0, 3, 1, 2)

class SparseAutoencoder(nn.Module):
    """Overcomplete ReLU SAE with unit-norm decoder columns (as in
    Anthropic's "Towards Monosemanticity" setup), trained per adapter
    layer on flattened per-pixel activations."""
 
    def __init__(self, d_in, d_hidden, l1_coeff=1e-3):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff
        self.W_enc = nn.Parameter(torch.randn(d_in, d_hidden) * (1.0 / math.sqrt(d_in)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.randn(d_hidden, d_in) * (1.0 / math.sqrt(d_hidden)))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
 
        # Raw VGG conv activations are unnormalized and often large/skewed,
        # which pushes ReLU((x-b_dec)@W_enc+b_enc) hard to one side and was
        # the main cause of the SAE collapsing to "mostly dead + a few
        # always-on" features. These buffers hold the per-channel mean/std
        # of the RAW activations the SAE was trained on; encode()/decode()
        # normalize on the way in and denormalize on the way out, so every
        # existing caller (hooks, diagnostics, sweep) keeps working in raw
        # activation units without any change to its own code.
        self.register_buffer("act_mean", torch.zeros(d_in))
        self.register_buffer("act_std", torch.ones(d_in))
 
    def set_norm_stats(self, mean, std):
        self.act_mean.copy_(mean.view(-1).to(self.act_mean.device))
        self.act_std.copy_(std.view(-1).clamp_min(1e-6).to(self.act_std.device))
 
    def normalize_decoder_(self):
        with torch.no_grad():
            norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
            self.W_dec.div_(norms)
 
    def encode(self, x):
        x_norm = (x - self.act_mean) / self.act_std
        return F.relu((x_norm - self.b_dec) @ self.W_enc + self.b_enc)
 
    def decode(self, feats):
        recon_norm = feats @ self.W_dec + self.b_dec
        return recon_norm * self.act_std + self.act_mean
 
    def forward(self, x):
        feats = self.encode(x)
        return self.decode(feats), feats
 
    def loss(self, x):
        recon, feats = self(x)
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        l1_loss = feats.abs().mean()
        return recon_loss + self.l1_coeff * l1_loss, recon_loss.item(), l1_loss.item(), feats
    

def collect_adapter_activations(model, layer_name, loader, max_batches=5, device=DEVICE):
    """Hook one adapter layer and gather its flattened per-pixel activations
    across a few batches, to use as SAE training data for that layer.
    Single-device version. `model` must be the UNWRAPPED module (not a
    DataParallel wrapper) so the hook fires reliably."""
    acts = []
 
    def hook(m, i, o):
        flat, _ = _flatten_spatial(o.detach())
        acts.append(flat.cpu())
 
    handle = model.adapters[layer_name].register_forward_hook(hook)
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            model(x.to(device))
            if i >= max_batches:
                break
    handle.remove()
    return torch.cat(acts, dim=0)


def collect_adapter_activations_multi_gpu(model, layer_name, loader, max_batches=5, devices=None):
    """
    Multi-GPU activation collection for SAE training data.
 
    WHY NOT nn.DataParallel: DataParallel replicates the whole module
    (hooks included) to every GPU on every forward call and runs those
    replicas in worker threads. The hook closures here write into a
    single shared Python list (`acts`), so concurrent replica threads can
    race on that write -- not safe. Instead, this function makes one
    REAL, independent copy of the model per GPU, gives each copy its own
    private accumulator list, and hands each copy a disjoint shard of
    batches. Each copy's hook only ever runs on its own GPU/thread, so
    there's no shared mutable state between them -- correct, and still
    gets real parallelism because CUDA ops release the GIL.
 
    This is the embarrassingly-parallel part of the pipeline (just
    forward passes to grab activations). The hook-based circuit tracing
    itself (AdapterCircuitTracer / SAECircuitTracer.compute_circuits)
    stays single-GPU -- ablation there is inherently sequential (each
    channel/feature ablation depends on comparing against a shared
    baseline) and isn't worth the correctness risk of parallelizing.
    """
    devices = devices or get_available_devices()
    if len(devices) == 1 or devices[0].type == "cpu":
        return collect_adapter_activations(model, layer_name, loader, max_batches, devices[0])
 
    # one real copy of the model per GPU
    replicas = [copy.deepcopy(model).to(d).eval() for d in devices]
 
    # grab up to max_batches+1 batches and round-robin them across devices
    batches = list(itertools.islice(loader, max_batches + 1))
    shards = {i: [] for i in range(len(devices))}
    for i, batch in enumerate(batches):
        shards[i % len(devices)].append(batch)
 
    def worker(replica, device, batch_list):
        local_acts = []
 
        def hook(m, i, o):
            flat, _ = _flatten_spatial(o.detach())
            local_acts.append(flat.cpu())
 
        handle = replica.adapters[layer_name].register_forward_hook(hook)
        with torch.no_grad():
            for x, y in batch_list:
                replica(x.to(device))
        handle.remove()
        return local_acts
 
    all_acts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as ex:
        futures = [ex.submit(worker, replicas[i], devices[i], shards[i])
                   for i in range(len(devices)) if shards[i]]
        for f in concurrent.futures.as_completed(futures):
            all_acts.extend(f.result())
 
    del replicas  # free GPU memory used by the extra copies
    return torch.cat(all_acts, dim=0)

def train_sae(activations, d_hidden_mult=8, l1_coeff=1e-3, epochs=200,
              lr=1e-3, batch_size=256, device=DEVICE, verbose=True,
              resample_every=25, resample_dead_thresh=0.0):
    """
    Trains the SAE, and periodically RESAMPLES dead features (a standard
    fix for the collapse you're seeing: most features dead, a handful
    always-on). Without this, high L1 + limited data very easily collapses
    into "most features silent, a few features absorb all the
    reconstruction burden and end up firing on everything" -- which is
    what the bimodal 0%/100% firing-rate histogram shows.
 
    Resampling: every `resample_every` epochs, find features with firing
    rate <= resample_dead_thresh over a fresh pass through the data, and
    re-initialize their encoder/decoder rows using actual data points that
    currently have the highest reconstruction error (so dead features get
    a fresh chance to specialize on inputs the SAE is failing to explain,
    instead of sitting at a dead initialization forever).
    """
    activations = activations.to(device)
    d_in = activations.shape[1]
    sae = SparseAutoencoder(d_in, d_in * d_hidden_mult, l1_coeff=l1_coeff).to(device)
 
    # Fix: set per-channel normalization stats from the RAW activations
    # before training. Without this, unnormalized VGG conv activations
    # (which can be large / skewed per channel) saturate the ReLU
    # encoder and collapse most features dead -- the failure mode you saw.
    act_mean = activations.mean(dim=0)
    act_std = activations.std(dim=0)
    sae.set_norm_stats(act_mean, act_std)
 
    sae.normalize_decoder_()
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    n = activations.shape[0]
 
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_recon, epoch_l1 = 0.0, 0.0
        for i in range(0, n, batch_size):
            batch = activations[perm[i:i + batch_size]]
            opt.zero_grad()
            total, recon_l, l1_l, _ = sae.loss(batch)
            total.backward()
            opt.step()
            sae.normalize_decoder_()
            epoch_recon += recon_l * batch.shape[0]
            epoch_l1 += l1_l * batch.shape[0]
 
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"[SAE {activations.shape[1]}d] epoch {epoch:4d}  "
                  f"recon={epoch_recon/n:.5f}  l1={epoch_l1/n:.5f}")
 
        if resample_every and (epoch + 1) % resample_every == 0 and epoch != epochs - 1:
            with torch.no_grad():
                recon, feats = sae(activations)
                firing_rate = (feats.abs() > 1e-6).float().mean(dim=0)
                dead_idx = (firing_rate <= resample_dead_thresh).nonzero(as_tuple=True)[0]
                if len(dead_idx) > 0:
                    err = (recon - activations).pow(2).sum(dim=1)
                    worst_idx = torch.topk(err, k=min(len(dead_idx), n)).indices
                    replacement_inputs = activations[worst_idx[:len(dead_idx)]]
                    # re-point dead decoder rows at the directions of the
                    # worst-reconstructed inputs, re-init matching encoder rows
                    new_dirs = replacement_inputs - sae.b_dec
                    new_dirs = new_dirs / new_dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    sae.W_dec.data[dead_idx] = new_dirs[:len(dead_idx)]
                    sae.W_enc.data[:, dead_idx] = new_dirs[:len(dead_idx)].T * 0.2
                    sae.b_enc.data[dead_idx] = 0.0
                    if verbose:
                        print(f"  resampled {len(dead_idx)} dead features at epoch {epoch}")
 
    return sae

class SAECircuitTracer:
    """Same role as AdapterCircuitTracer, but ablates SAE feature
    dimensions instead of raw conv channels. Ablation = encode per-pixel
    -> zero the feature everywhere -> decode -> reshape back to
    [B,C,H,W] -> substitute into the VGG forward pass."""
 
    def __init__(self, model, saes, device=DEVICE):
        self.model = model
        self.saes = saes  # {layer_name: trained SparseAutoencoder}
        self.device = device
 
    def _encode_cache(self, raw_cache):
        feat_cache = {}
        for layer, act in raw_cache.items():
            if layer not in self.saes:
                continue
            flat, _ = _flatten_spatial(act)
            feat_cache[layer] = self.saes[layer].encode(flat)
        return feat_cache
 
    def run_with_cache(self, x):
        raw_cache, hooks = {}, []
 
        def hook_fn(name):
            def fn(m, i, o):
                raw_cache[name] = o.detach()
            return fn
 
        for name, module in self.model.adapters.items():
            hooks.append(module.register_forward_hook(hook_fn(name)))
        out = self.model(x)
        for h in hooks:
            h.remove()
 
        return out, raw_cache, self._encode_cache(raw_cache)
 
    def patch_and_run(self, x, layer_name, feat_idx):
        raw_cache, hooks = {}, []
        sae = self.saes[layer_name]
 
        def patch_hook(m, i, o):
            flat, shp = _flatten_spatial(o)
            feats = sae.encode(flat).clone()
            feats[:, feat_idx] = 0.0
            recon_flat = sae.decode(feats)
            return _unflatten_spatial(recon_flat, shp, flat.shape[1])
 
        def save_hook(name):
            def fn(m, i, o):
                raw_cache[name] = o.detach()
            return fn
 
        patch_handle = self.model.adapters[layer_name].register_forward_hook(patch_hook)
        for name, module in self.model.adapters.items():
            if name != layer_name:
                hooks.append(module.register_forward_hook(save_hook(name)))
 
        out = self.model(x)
 
        patch_handle.remove()
        for h in hooks:
            h.remove()
 
        return out, raw_cache, self._encode_cache(raw_cache)
 
    def compute_circuits(self, loader, max_batches=3, topk=TOPK):
        circuits = {}
 
        batch_bar = tqdm(enumerate(loader), total=max_batches + 1, desc="SAE circuit batches", position=0)
        for i, (x, y) in batch_bar:
            x, y = x.to(self.device), y.to(self.device)
            out, raw_cache, feat_cache = self.run_with_cache(x)
 
            classes = y.unique().tolist()
            class_bar = tqdm(classes, desc=f"batch {i} classes", position=1, leave=False)
            for cls in class_bar:
                idx = (y == cls)
                base = out[idx, cls].mean()
 
                if cls not in circuits:
                    circuits[cls] = nx.DiGraph()
 
                layer_topk = {}
                for layer, feats in feat_cache.items():
                    scores = []
                    feat_bar = tqdm(range(feats.shape[1]), desc=f"cls {cls} layer {layer} features",
                                     position=2, leave=False)
                    for f in feat_bar:
                        patched_out, _, _ = self.patch_and_run(x, layer, f)
                        eff = (base - patched_out[idx, cls].mean()).item()
                        scores.append((f, eff))
                    scores = sorted(scores, key=lambda s: abs(s[1]), reverse=True)[:topk]
                    layer_topk[layer] = scores
                    for f, eff in scores:
                        node = f"{layer}_feat{f}"
                        circuits[cls].add_node(node, importance=eff, layer=layer, feat_idx=f)
 
                layers = list(feat_cache.keys())
                edge_pairs = list(zip(layers[:-1], layers[1:]))
                edge_bar = tqdm(edge_pairs, desc=f"cls {cls} edges", position=2, leave=False)
                for l1, l2 in edge_bar:
                    for f1, _ in layer_topk[l1]:
                        _, _, patched_feat_cache = self.patch_and_run(x, l1, f1)
                        for f2, _ in layer_topk[l2]:
                            base_tgt = feat_cache[l2][:, f2].mean()
                            patched_tgt = patched_feat_cache[l2][:, f2].mean()
                            eff = (base_tgt - patched_tgt).item()
                            if abs(eff) > 0.001:
                                circuits[cls].add_edge(f"{l1}_feat{f1}", f"{l2}_feat{f2}", weight=eff)
 
            if i >= max_batches:
                break
 
        return circuits
    
# -----------------------------
# MONOSEMANTICITY DIAGNOSTICS (evidence, not proof)
# -----------------------------
def class_conditional_entropy(feat_activations, labels, n_classes, eps=1e-8):
    """Per feature: entropy of activation mass across classes.
    Low = fires almost exclusively for one class (supports monosemanticity).
    High = fires broadly across classes.
 
    FIXED: a feature that never fires has zero mass for every class. The
    old version divided 0/0 -> 0 everywhere -> entropy 0 -> looked like
    the MOST class-selective (best) feature, when it's actually just off.
    Dead features are now returned as NaN so callers can (and must)
    explicitly exclude them instead of silently ranking them first.
    """
    if feat_activations.ndim != 2:
        raise ValueError("feat_activations must be 2D [N, F]")
    labels = labels.reshape(-1)
    if labels.shape[0] != feat_activations.shape[0]:
        n = min(labels.shape[0], feat_activations.shape[0])
        print(
            f"Aligning entropy inputs: labels={labels.shape[0]} "
            f"activations={feat_activations.shape[0]} -> using {n} rows"
        )
        labels = labels[:n]
        feat_activations = feat_activations[:n]

    labels = labels.to(device=feat_activations.device)
    Fh = feat_activations.shape[1]
    mass = torch.zeros(Fh, n_classes, device=feat_activations.device)
    for c in range(n_classes):
        idx = (labels == c)
        if idx.sum() > 0:
            mass[:, c] = feat_activations[idx].clamp_min(0).sum(dim=0)
 
    total = mass.sum(dim=1)  # [Fh]
    alive = total > eps
 
    ent = torch.full((Fh,), float("nan"), device=feat_activations.device)
    if alive.any():
        p = mass[alive] / total[alive].unsqueeze(1)
        ent_alive = -(p * p.clamp_min(eps).log()).sum(dim=1)
        ent[alive] = ent_alive / math.log(n_classes) if n_classes > 1 else ent_alive
    return ent  # NaN entries = dead features; exclude before ranking/plotting

def activation_sparsity(feat_activations, threshold=1e-6):
    active = (feat_activations.abs() > threshold).float()
    overall_sparsity = 1.0 - active.mean().item()
    per_feature_firing_rate = active.mean(dim=0)
    dead_features = (per_feature_firing_rate == 0).sum().item()
    return overall_sparsity, per_feature_firing_rate, dead_features

def top_activating_examples(feat_activations, feat_idx, labels, k=8):
    """Indices/labels of the top-k examples (pixels, in this conv setting)
    that most activate a feature -- for manual inspection."""
    col = feat_activations[:, feat_idx]
    topk = torch.topk(col, k=min(k, col.shape[0]))
    return {"indices": topk.indices.tolist(), "activations": topk.values.tolist(),
            "labels": labels[topk.indices].tolist() if labels is not None else None}

def decoder_feature_similarity(sae, feat_indices):
    """Cosine similarity between decoder directions -- near-duplicates
    across features flag feature splitting, not confirm monosemanticity."""
    W = sae.W_dec[feat_indices]
    return W @ W.T

def sae_hyperparameter_sweep(activations, labels, n_classes,
                              l1_coeffs=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
                              d_hidden_mults=(4, 8), epochs=150, lr=1e-3,
                              batch_size=256, device=DEVICE, verbose=False):
    """Vary the degree of monosemanticity by retraining across a grid of
    (l1_coeff, d_hidden_mult). There is no single 'most monosemantic'
    point -- only a fidelity/selectivity tradeoff curve."""
    results = []
    activations = activations.to(device)
    labels = labels.to(device)
 
    for mult in d_hidden_mults:
        for l1 in l1_coeffs:
            sae = train_sae(activations, d_hidden_mult=mult, l1_coeff=l1,
                             epochs=epochs, lr=lr, batch_size=batch_size,
                             device=device, verbose=verbose)
            with torch.no_grad():
                recon, feats = sae(activations)
                recon_mse = F.mse_loss(recon, activations).item()
            ent = class_conditional_entropy(feats, labels, n_classes)
            overall_sparsity, firing_rate, dead = activation_sparsity(feats)
 
            # NaN-aware: dead features are excluded from entropy stats
            # entirely, rather than silently pulling mean/median down to
            # a falsely great-looking score.
            alive_ent = ent[~torch.isnan(ent)]
            mean_entropy = alive_ent.mean().item() if alive_ent.numel() > 0 else float("nan")
            median_entropy = alive_ent.median().item() if alive_ent.numel() > 0 else float("nan")
 
            results.append({
                "l1_coeff": l1, "d_hidden_mult": mult, "sae": sae,
                "recon_mse": recon_mse,
                "mean_entropy": mean_entropy,
                "median_entropy": median_entropy,
                "overall_sparsity": overall_sparsity,
                "dead_feature_frac": dead / feats.shape[1],
            })
            print(f"[sweep] mult={mult} l1={l1:.0e}  recon_mse={recon_mse:.5f}  "
                  f"mean_entropy(alive only)={mean_entropy:.3f}  dead_frac={dead/feats.shape[1]:.2f}")
    return results

# -----------------------------
# VISUALIZATIONS
# -----------------------------
def plot_class_entropy_histogram(ent_norm, save_path=None):
    vals = ent_norm.detach().cpu().numpy()
    dead_count = np.isnan(vals).sum()
    alive_vals = vals[~np.isnan(vals)]
 
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(alive_vals, bins=40, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Normalized class-conditional entropy (0=class-exclusive, 1=uniform)")
    ax.set_ylabel("# SAE features")
    ax.set_title(f"Class selectivity of SAE features (alive only)\n"
                 f"(low entropy = supports monosemanticity; {dead_count} dead features excluded)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_feature_class_heatmap(feat_activations, labels, n_classes, feat_indices, save_path=None):
    mat = np.zeros((len(feat_indices), n_classes))
    for row, f in enumerate(feat_indices):
        for c in range(n_classes):
            idx = (labels == c)
            if idx.sum() > 0:
                mat[row, c] = feat_activations[idx, f].clamp_min(0).mean().item()
    fig, ax = plt.subplots(figsize=(max(4, n_classes * 0.3), max(3, len(feat_indices) * 0.35)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(feat_indices)))
    ax.set_yticklabels([f"feat {f}" for f in feat_indices])
    ax.set_xlabel("Class")
    ax.set_title("Mean feature activation by class\n(one bright column per row = class-selective)")
    fig.colorbar(im, ax=ax, label="mean activation")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig

def plot_dead_and_firing_rate(per_feature_firing_rate, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    rates = per_feature_firing_rate.detach().cpu().numpy()
    ax.hist(rates, bins=40, color="#55A868", edgecolor="white")
    ax.set_xlabel("Firing rate (fraction of examples that activate this feature)")
    ax.set_ylabel("# SAE features")
    ax.set_title(f"SAE feature firing rates\n({(rates==0).sum()} dead features)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig

def plot_decoder_similarity(sim_matrix, feat_indices, save_path=None):
    fig, ax = plt.subplots(figsize=(max(4, len(feat_indices) * 0.4), max(4, len(feat_indices) * 0.4)))
    im = ax.imshow(sim_matrix.detach().cpu().numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feat_indices))); ax.set_xticklabels(feat_indices, rotation=90)
    ax.set_yticks(range(len(feat_indices))); ax.set_yticklabels(feat_indices)
    ax.set_title("Decoder direction cosine similarity\n(off-diagonal near ±1 = feature splitting risk)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig

def plot_monosemanticity_tradeoff(sweep_results, save_path=None):
    """The 'varying the degree of monosemanticity' picture: entropy
    (selectivity) vs. reconstruction fidelity across swept configs.
    Point size = dead-feature fraction, to catch false wins from a
    mostly-dead SAE."""
    fig, ax = plt.subplots(figsize=(7, 5))
    mults = sorted(set(r["d_hidden_mult"] for r in sweep_results))
    cmap = plt.get_cmap("tab10")
    for i, mult in enumerate(mults):
        pts = [r for r in sweep_results if r["d_hidden_mult"] == mult]
        xs = [r["mean_entropy"] for r in pts]
        ys = [r["recon_mse"] for r in pts]
        sizes = [40 + 400 * r["dead_feature_frac"] for r in pts]
        ax.scatter(xs, ys, s=sizes, color=cmap(i), alpha=0.7, label=f"dict size x{mult}")
        for r in pts:
            ax.annotate(f"l1={r['l1_coeff']:.0e}", (r["mean_entropy"], r["recon_mse"]),
                        fontsize=7, alpha=0.7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean class-conditional entropy (lower = more class-selective)")
    ax.set_ylabel("Reconstruction MSE (lower = more faithful to true activations)")
    ax.set_title("Monosemanticity/fidelity tradeoff across SAE configs\n(point size = dead-feature fraction)")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig

# =====================================================================
# EXAMPLE RUN: SAE + SAE-circuit + diagnostics on the Art-domain model
# =====================================================================
if torch.cuda.is_available():
    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(i) for i in range(torch.cuda.device_count())
        )

model_for_tracing = base_module(model)
DEVICES = get_available_devices()
print(f"\n=== SAE TRAINING (Art domain, layer 28) === using devices: {DEVICES}")
LAYER_FOR_SAE = "28"  # pick one adapter layer to demonstrate on; repeat per layer as needed
SAE_MAX_BATCHES = 1  # must match the batch count used when collecting pixel labels below

# Activation collection is the embarrassingly-parallel part -- shard it
# across all visible GPUs. `model_for_tracing` is the UNWRAPPED module
# (see caveat near its definition): required for forward hooks to fire
# reliably, and it's what gets deepcopy'd onto each GPU below.
acts_28 = collect_adapter_activations_multi_gpu(
    model_for_tracing,
    LAYER_FOR_SAE,
    train_loader,
    max_batches=SAE_MAX_BATCHES,
    devices=DEVICES,
)

# labels aligned with the collected activations (per-pixel repeats of the
# image-level label), needed for the class-conditional diagnostics
def collect_pixel_labels(model, layer_name, loader, max_batches=5, device=DEVICE):
    labels_out = []
    last_shape = None

    def hook(m, i, o):
        nonlocal last_shape
        last_shape = o.shape  # (B,C,H,W)

    handle = model.adapters[layer_name].register_forward_hook(hook)
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            model(x.to(device))
            if last_shape is None:
                continue
            B, C, H, W = last_shape
            labels_out.append(y.repeat_interleave(H * W))
            if i >= max_batches:
                break
    handle.remove()
    return torch.cat(labels_out, dim=0)

# Pixel labels must line up 1:1 with the batches actually consumed by
# acts_28. Since collect_adapter_activations_multi_gpu round-robins the
# SAME `max_batches+1` batches drawn from `loader` in order, redraw the
# identical sequence here (single device is fine -- this is cheap, it's
# just forwarding to read shapes, not the expensive SAE work).
pixel_labels_28 = collect_pixel_labels(
    model_for_tracing,
    LAYER_FOR_SAE,
    train_loader,
    max_batches=SAE_MAX_BATCHES,
)

if pixel_labels_28.shape[0] != acts_28.shape[0]:
    n = min(pixel_labels_28.shape[0], acts_28.shape[0])
    print(
        f"Aligning labels to activations: labels={pixel_labels_28.shape[0]} "
        f"activations={acts_28.shape[0]} -> using {n} rows"
    )
    pixel_labels_28 = pixel_labels_28[:n]
    acts_28 = acts_28[:n]

# 1) sweep to see the monosemanticity/fidelity tradeoff and pick a config
sweep_results = sae_hyperparameter_sweep(
    acts_28, pixel_labels_28, n_classes=NUM_CLASSES,
    l1_coeffs=(1e-4, 1e-3, 1e-2), d_hidden_mults=(4, 8), epochs=100,
)

plot_monosemanticity_tradeoff(sweep_results, save_path=os.path.join(SAVE_DIR, "tradeoff_layer28.png"))

# 2) pick the config with lowest ALIVE-feature entropy that still has a
#    low dead-feature fraction (a defensible choice, not "the maximum").
#    Configs whose mean_entropy came back NaN (i.e. almost everything
#    died) are excluded outright rather than being able to win by default.
candidates = [r for r in sweep_results
              if r["dead_feature_frac"] < 0.3 and not math.isnan(r["mean_entropy"])]
if not candidates:
    candidates = [r for r in sweep_results if not math.isnan(r["mean_entropy"])]
best = min(candidates, key=lambda r: r["mean_entropy"])
print(f"Selected config: l1={best['l1_coeff']:.0e}, dict_mult={best['d_hidden_mult']}, "
      f"mean_entropy(alive only)={best['mean_entropy']:.3f}, dead_frac={best['dead_feature_frac']:.2f}")
sae_28 = best["sae"]

# 3) diagnostics for the selected SAE
with torch.no_grad():
    _, feats_28 = sae_28(acts_28.to(DEVICE))
ent_28 = class_conditional_entropy(feats_28, pixel_labels_28.to(DEVICE), NUM_CLASSES)
overall_sparsity, firing_rate, dead = activation_sparsity(feats_28)
plot_class_entropy_histogram(ent_28, save_path=os.path.join(SAVE_DIR, "entropy_layer28.png"))
plot_dead_and_firing_rate(firing_rate, save_path=os.path.join(SAVE_DIR, "firing_rate_layer28.png"))

# Rank only ALIVE features by entropy -- dead ones are NaN and must be
# filtered out first, otherwise argsort puts NaNs at the end/undefined
# position but you could still accidentally include one at the boundary.
alive_idx = (~torch.isnan(ent_28)).nonzero(as_tuple=True)[0]
if alive_idx.numel() == 0:
    print("WARNING: every feature in this SAE is dead. Loosen l1_coeff / "
          "increase dict size / check normalization before trusting anything else here.")
    top_features = []
else:
    ranked = alive_idx[torch.argsort(ent_28[alive_idx])]
    top_features = ranked[:min(20, ranked.numel())].tolist()
    plot_feature_class_heatmap(feats_28, pixel_labels_28.to(DEVICE), NUM_CLASSES, top_features,
                                save_path=os.path.join(SAVE_DIR, "heatmap_layer28.png"))
    sim = decoder_feature_similarity(sae_28, top_features)
    plot_decoder_similarity(sim, top_features, save_path=os.path.join(SAVE_DIR, "decoder_sim_layer28.png"))
 
print(f"Saved SAE diagnostic plots to {SAVE_DIR}")

# 4) train SAEs for all adapter layers (each layer's activation collection
#    sharded across GPUs), then build the SAE-based circuit
saes = {LAYER_FOR_SAE: sae_28}
for layer_name in model_for_tracing.adapters.keys():
    if layer_name == LAYER_FOR_SAE:
        continue
    acts_l = collect_adapter_activations_multi_gpu(
        model_for_tracing, layer_name, train_loader, max_batches=5, devices=DEVICES
    )
    saes[layer_name] = train_sae(acts_l, d_hidden_mult=best["d_hidden_mult"],
                                  l1_coeff=best["l1_coeff"], epochs=100)
    

import pickle
import networkx as nx
from pyvis.network import Network
print("\n=== SAE CIRCUIT EXTRACTION (Art domain) ===")
# NOTE: stays single-GPU on purpose -- see the caveat on
# collect_adapter_activations_multi_gpu / model_for_tracing above.
sae_tracer = SAECircuitTracer(model_for_tracing, saes)
sae_circuits = sae_tracer.compute_circuits(train_loader, max_batches=1)  # feature-level tracing is expensive; keep batches small
 
sae_circuit_path = os.path.join(SAVE_DIR, "sae_circuits_vgg_art.pkl")
with open(sae_circuit_path, "wb") as f:
    pickle.dump(sae_circuits, f)
print(f"✅ SAE-based circuits saved at {sae_circuit_path}")
 
cls0 = list(sae_circuits.keys())[0]
print(f"Class {cls0} SAE circuit: {sae_circuits[cls0]}")
#visualize_layered(sae_circuits[cls0], file=os.path.join(SAVE_DIR, "sae_circuit_layered.html"))


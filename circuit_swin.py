import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import networkx as nx
from pyvis.network import Network

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_CLASSES = 345
EPOCHS = 10000
TOPK = 5
PATIENCE = 30
SAVE_DIR = "./outputs_domainnet"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Swin-T stage layout (torchvision swin_t):
#
#   swin.features index | content              | output C dim
#   --------------------+----------------------+--------------
#   0                   | patch embed          | 96
#   1                   | stage-0 (2 blocks)   | 96
#   2                   | PatchMerging         | 192
#   3                   | stage-1 (2 blocks)   | 192
#   4                   | PatchMerging         | 384
#   5                   | stage-2 (6 blocks)   | 384
#   6                   | PatchMerging         | 768
#   7                   | stage-3 (2 blocks)   | 768
#
# After features: swin.norm -> swin.permute -> swin.avgpool -> swin.flatten
# ------------------------------------------------------------------
SWIN_STAGE_DIMS  = {1: 96, 3: 192, 5: 384, 7: 768}   # C for every stage feature index
ADAPTER_FEAT_IDX = [3, 5, 7]                            # mirror ViT's layers [8,10,11]
#
# ⚠️  Swin vs ViT tensor shapes
#   ViT adapter: (B, N, D)  — N = seq length incl. CLS token
#   Swin adapter: (B, H, W, C) — 2-D spatial feature map
#
# Consequence: every place the ViT code indexed [:,0,dim] (CLS token)
# we use [:,:,:,dim] (all spatial positions) and average over H,W for
# scalar importance scores.

# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------
class SwinAdapter(nn.Module):
    """
    Bottleneck residual adapter for one Swin stage.

    Input/output shape: (B, H, W, C)
    Identical algebra to ViTAdapter but the tensor is 4-D not 3-D;
    nn.Linear acts on the last dimension, so no reshape is required.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        return x + self.mlp(x)


class Swin_Adapter(nn.Module):
    """
    Swin-T with frozen backbone + lightweight adapters + new head.

    Key differences from ViT_Adapter
    ---------------------------------
    1. No fix_vit_encoder() needed: SwinTransformerBlock.forward() and
       ShiftedWindowAttention.forward() both return plain Tensors, not tuples.

    2. The CLS-token trick doesn't apply to Swin.  The global representation
       is produced by swin.avgpool after the full feature stack, so we call
       swin.norm / permute / avgpool / flatten explicitly instead of indexing
       a CLS position.

    3. Each stage has a different C dimension (96/192/384/768), so adapters
       must receive the right dim for their stage.
    """
    def __init__(self, num_classes: int = 345):
        super().__init__()
        self.swin = models.swin_t(weights="IMAGENET1K_V1")

        # Freeze the entire Swin backbone (same strategy as ViT_Adapter)
        for p in self.swin.parameters():
            p.requires_grad = False

        self.adapter_feat_indices = ADAPTER_FEAT_IDX

        # One adapter per selected stage; keyed by the features[] index (as str)
        self.adapters = nn.ModuleDict({
            str(i): SwinAdapter(SWIN_STAGE_DIMS[i])
            for i in self.adapter_feat_indices
        })

        # New classification head (replaces swin.head which targets 1000 classes)
        self.head = nn.Linear(SWIN_STAGE_DIMS[7], num_classes)   # 768 -> num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Walk features manually so we can inject adapters after chosen stages
        for i, layer in enumerate(self.swin.features):
            x = layer(x)                               # (B, H, W, C)
            if i in self.adapter_feat_indices:
                x = self.adapters[str(i)](x)           # residual adapter

        # Standard Swin head pipeline
        x = self.swin.norm(x)        # LayerNorm over C              (B, H, W, C)
        x = self.swin.permute(x)     # Permute([0,3,1,2])            (B, C, H, W)
        x = self.swin.avgpool(x)     # AdaptiveAvgPool2d(1)          (B, C, 1, 1)
        x = self.swin.flatten(x)     # Flatten(1)                    (B, C)
        return self.head(x)          # Linear                        (B, num_classes)


# No fix_encoder() equivalent needed for Swin — see docstring above.

model = Swin_Adapter(NUM_CLASSES).to(DEVICE)

# ------------------------------------------------------------------
# TRACER
# ------------------------------------------------------------------
def get_swin_attn_modules(model: Swin_Adapter) -> dict:
    """
    Return every ShiftedWindowAttention module keyed as "f{feat_idx}_b{block_idx}".

    Example keys: "f1_b0", "f1_b1", "f3_b0", ..., "f7_b1"
    Total for swin_t: 2 + 2 + 6 + 2 = 12 attention modules.

    ViT equivalent: {str(i): block.self_attention for i, block in enumerate(encoder.layers)}
    Swin equivalent needs two-level iteration (stage → block).
    """
    attn_modules: dict = {}
    for feat_idx in SWIN_STAGE_DIMS:                        # 1, 3, 5, 7
        stage = model.swin.features[feat_idx]               # Sequential[SwinTransformerBlock]
        for block_idx, block in enumerate(stage):
            key = f"f{feat_idx}_b{block_idx}"
            attn_modules[key] = block.attn                  # ShiftedWindowAttention
    return attn_modules


def _attn_sort_key(k: str):
    """Sort "f5_b3" -> (5, 3) so layers are in forward-pass order."""
    feat, blk = k.split("_")
    return (int(feat[1:]), int(blk[1:]))


class CircuitTracer:
    def __init__(self, model: Swin_Adapter):
        self.model = model
        self.attn = get_swin_attn_modules(model)

    # ------------------------------------------------------------------
    def run_with_cache(self, x: torch.Tensor):
        """
        Forward pass that stores adapter and attention outputs in `cache`.

        Cache shapes (Swin):
          "mlp_{layer}"  -> (B, H, W, C)   adapter output
          "attn_{key}"   -> (B, H, W, C)   attention output

        ViT equivalent stored (B, N, D) for both; Swin just has 4-D instead of 3-D.
        get_tensor() unwrapping is NOT needed here because Swin returns plain Tensors.
        """
        cache: dict = {}
        hooks: list = []

        def save(name: str):
            def fn(m, i, o):
                cache[name] = o.detach()    # (B, H, W, C)
            return fn

        for n, m in self.model.adapters.items():
            hooks.append(m.register_forward_hook(save("mlp_" + n)))

        for n, m in self.attn.items():
            hooks.append(m.register_forward_hook(save("attn_" + n)))

        out = self.model(x)

        for h in hooks:
            h.remove()

        return out, cache

    # ------------------------------------------------------------------
    def patch_mlp(self, x: torch.Tensor, layer: str, dim: int):
        """
        Zero out channel `dim` in adapter `layer`'s output.

        ViT:  o[:, 0, dim] = 0   (CLS token position, 3-D tensor)
        Swin: o[:, :, :, dim] = 0  (all spatial positions, 4-D tensor)
        """
        def hook(m, i, o):
            o = o.clone()
            o[:, :, :, dim] = 0
            return o

        h = self.model.adapters[layer].register_forward_hook(hook)
        out, cache = self.run_with_cache(x)
        h.remove()
        return out, cache

    # ------------------------------------------------------------------
    def patch_head(self, x: torch.Tensor, layer: str, head: int):
        """
        Zero out the contribution of attention head `head` in block `layer`.

        ShiftedWindowAttention returns (B, H, W, C) where heads are packed
        along the C dimension (identical layout to ViT's (B, N, D) after
        self_attention, just with an extra spatial dimension H,W).

        ViT:  o = o.view(B, N, H, Hd); o[:,:,head] = 0; o.view(B, N, D)
        Swin: zero the head's slice along C — no reshape needed because
              the spatial dims are independent of the head split.
        """
        attn_mod = self.attn[layer]
        num_heads = attn_mod.num_heads   # swin_t per stage: 3, 6, 12, 24

        def hook(m, i, o):
            o = o.clone()
            B, H, W, C = o.shape
            Hd = C // num_heads
            o[:, :, :, head * Hd : (head + 1) * Hd] = 0
            return o

        h = attn_mod.register_forward_hook(hook)
        out, cache = self.run_with_cache(x)
        h.remove()
        return out, cache

    # ------------------------------------------------------------------
    def compute(self, loader: DataLoader, max_batches: int = 1) -> dict:
        """
        Build per-class circuit graphs via activation patching.

        Importance scores are logit-difference: base_logit - patched_logit.

        The only Swin-specific change vs the ViT version is in edge
        comparison: ViT used cache["attn_"+l2][:,0,:].mean() (CLS token),
        Swin uses cache["attn_"+l2].mean() (global spatial average).
        """
        circuits: dict = {}

        for i, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out, cache = self.run_with_cache(x)

            for cls in y.unique():
                cls = cls.item()
                idx = (y == cls)
                base = out[idx, cls].mean()

                if cls not in circuits:
                    circuits[cls] = nx.DiGraph()
                G = circuits[cls]

                mlp_top: dict = {}
                attn_top: dict = {}

                # ---- Adapter (MLP) nodes ----------------------------------------
                for layer in self.model.adapters:
                    D = cache["mlp_" + layer].shape[-1]    # C dim

                    scores = []
                    for d in torch.randperm(D)[:TOPK]:
                        patched, _ = self.patch_mlp(x, layer, d.item())
                        eff = (base - patched[idx, cls].mean()).item()
                        scores.append((d.item(), eff))

                    scores.sort(key=lambda z: abs(z[1]), reverse=True)
                    mlp_top[layer] = scores[:TOPK]

                    for d, eff in mlp_top[layer]:
                        G.add_node(f"{layer}_mlp_{d}", layer=layer, importance=eff)

                # ---- Attention nodes --------------------------------------------
                for layer in self.attn:
                    num_heads = self.attn[layer].num_heads

                    scores = []
                    for h in range(num_heads):
                        patched, _ = self.patch_head(x, layer, h)
                        eff = (base - patched[idx, cls].mean()).item()
                        scores.append((h, eff))

                    scores.sort(key=lambda z: abs(z[1]), reverse=True)
                    attn_top[layer] = scores[:TOPK]

                    for h, eff in attn_top[layer]:
                        G.add_node(f"{layer}_head_{h}", layer=layer, importance=eff)

                # ---- Edges -------------------------------------------------------
                attn_layers = sorted(self.attn.keys(), key=_attn_sort_key)
                adapter_layers = sorted(self.model.adapters.keys(), key=int)

                # Attention -> Attention (consecutive Swin blocks)
                for l1, l2 in zip(attn_layers[:-1], attn_layers[1:]):
                    for h1, _ in attn_top.get(l1, []):
                        _, patched_cache = self.patch_head(x, l1, h1)

                        for h2, _ in attn_top.get(l2, []):
                            # Swin: spatial mean instead of ViT's CLS [:,0,:].mean()
                            eff = (
                                cache["attn_" + l2].mean()
                                - patched_cache["attn_" + l2].mean()
                            ).item()

                            if abs(eff) > 1e-4:
                                G.add_edge(
                                    f"{l1}_head_{h1}", f"{l2}_head_{h2}", weight=eff
                                )

                # Adapter -> Adapter (consecutive adapter stages)
                for l1, l2 in zip(adapter_layers[:-1], adapter_layers[1:]):
                    for d1, _ in mlp_top.get(l1, []):
                        _, patched_cache = self.patch_mlp(x, l1, d1)

                        for d2, _ in mlp_top.get(l2, []):
                            # Swin: mean over (B, H, W) for channel d2
                            eff = (
                                cache["mlp_" + l2][:, :, :, d2].mean()
                                - patched_cache["mlp_" + l2][:, :, :, d2].mean()
                            ).item()

                            if abs(eff) > 1e-4:
                                G.add_edge(
                                    f"{l1}_mlp_{d1}", f"{l2}_mlp_{d2}", weight=eff
                                )

            if i >= max_batches:
                break

        return circuits


# ------------------------------------------------------------------
# TRAINING  — Sketch domain  (mirrors original active section)
# ------------------------------------------------------------------
dataset_sketch = datasets.ImageFolder(
    "/home/aban/circuit_tracing/DomainNet/sketch", transform=transform
)

generator_sketch = torch.Generator().manual_seed(42)
train_size_sketch = int(0.8 * len(dataset_sketch))
test_size_sketch  = len(dataset_sketch) - train_size_sketch

train_dataset_sketch, test_dataset_sketch = random_split(
    dataset_sketch, [train_size_sketch, test_size_sketch], generator=generator_sketch
)

train_loader_sketch = DataLoader(
    train_dataset_sketch, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)
test_loader_sketch = DataLoader(
    test_dataset_sketch, batch_size=BATCH_SIZE, num_workers=4
)

model_sketch = Swin_Adapter(NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model_sketch.parameters()), lr=1e-3
)

best_val = 0
patience_counter = 0


def train_epoch():
    model_sketch.train()
    total, correct = 0, 0
    for xb, yb in train_loader_sketch:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = model_sketch(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        correct += (out.argmax(1) == yb).sum().item()
        total   += yb.size(0)
    return correct / total


@torch.no_grad()
def evaluate():
    model_sketch.eval()
    total, correct = 0, 0
    for xb, yb in test_loader_sketch:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out = model_sketch(xb)
        correct += (out.argmax(1) == yb).sum().item()
        total   += yb.size(0)
    return correct / total


print("=== TRAINING ===")
for e in range(EPOCHS):
    train_acc = train_epoch()
    val_acc   = evaluate()
    print(f"Epoch {e+1}: Train {train_acc:.3f} | Val {val_acc:.3f}")

    if val_acc > best_val:
        best_val = val_acc
        patience_counter = 0
        torch.save(
            model_sketch.state_dict(),
            os.path.join(SAVE_DIR, "best_model_sketch_swin.pth")
        )
        print("✅ Model saved")
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("Early stopping")
        break

# ------------------------------------------------------------------
# LOAD BEST MODEL
# ------------------------------------------------------------------
model_sketch.load_state_dict(
    torch.load(os.path.join(SAVE_DIR, "best_model_sketch_swin.pth"))
)
model_sketch.eval()

# ------------------------------------------------------------------
# CIRCUIT EXTRACTION
# ------------------------------------------------------------------
print("\n=== CIRCUIT EXTRACTION ===")
tracer_sketch_swin   = CircuitTracer(model_sketch)
circuits_sketch_swin = tracer_sketch_swin.compute(train_loader_sketch)

# ------------------------------------------------------------------
# SAVE CIRCUITS
# ------------------------------------------------------------------
import pickle
circuit_path = os.path.join(SAVE_DIR, "circuits_swin_sketch.pkl")
with open(circuit_path, "wb") as f:
    pickle.dump(circuits_sketch_swin, f)

print(f"✅ Circuits saved at {circuit_path}")

# ------------------------------------------------------------------
# QUICK VISUALISATION  (same as original visualize_layered)
# ------------------------------------------------------------------
def visualize_layered(graph: nx.DiGraph, file: str = "circuit_layered_swin.html"):
    net = Network(directed=True)

    layer_groups: dict = {}
    for node, data in graph.nodes(data=True):
        layer = data.get("layer", "unknown")
        layer_groups.setdefault(layer, []).append(node)

    x_gap, y_gap = 200, 50
    for i, (layer, nodes) in enumerate(sorted(layer_groups.items())):
        for j, node in enumerate(nodes):
            net.add_node(
                node,
                x=i * x_gap,
                y=j * y_gap,
                fixed=True,
                value=abs(graph.nodes[node].get("importance", 1)),
                title=f"Layer {layer}",
            )

    for u, v, data in graph.edges(data=True):
        net.add_edge(u, v, value=abs(data.get("weight", 1)))

    net.write_html(file)
    print(f"Saved visualization to {file}")


cls = list(circuits_sketch_swin.keys())[0]
print(cls)
print(circuits_sketch_swin[cls])
# visualize_layered(circuits_sketch_swin[cls])

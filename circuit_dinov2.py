import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import networkx as nx
from pyvis.network import Network

import multiprocessing
multiprocessing.set_start_method("fork", force=True)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE      = 64
NUM_CLASSES     = 7
EPOCHS          = 10000
TOPK            = 5
PATIENCE        = 30
SAVE_DIR        = "./outputs_pacs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------------------------
# DINOv2 architecture constants  (ViT-B/14 defaults)
# ------------------------------------------------------------------
DINO_MODEL_NAME     = "dinov2_vitb14"
#
# torch.hub model name -> embed_dim / num_heads / num_blocks
#   dinov2_vits14  ->  384 /  6 / 12   (adapter layers [8,10,11])
#   dinov2_vitb14  ->  768 / 12 / 12   (adapter layers [8,10,11])  <-- default
#   dinov2_vitl14  -> 1024 / 16 / 24   (adapter layers [20,22,23])
#   dinov2_vitg14  -> 1536 / 24 / 40   (adapter layers [36,38,39])
#
# Register variants (dinov2_vitb14_reg, etc.) add extra register tokens
# between CLS and patch tokens but do NOT change the embedding dim or
# the block structure -- the code below works with them unchanged.
#
DINO_ADAPTER_LAYERS = [8, 10, 11]   # last-3-block pattern from original ViT code

# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
# DINOv2 uses standard ImageNet normalisation -- same stats as the
# original ViT code. No change needed here.
# Patch size is 14, so 224x224 gives a 16x16 patch grid (256 patches).
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------
class DINOAdapter(nn.Module):
    """
    Bottleneck residual adapter for a DINOv2 transformer block.

    Input/output shape: (B, N, D) -- batch-first, identical to the
    original ViT code.  nn.Linear operates on the last dim D, so no
    reshape is needed.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        return x + self.mlp(x)


class DINOv2_Adapter(nn.Module):
    """
    DINOv2 ViT backbone (frozen) + lightweight adapters + new classifier head.

    Comparison with the three previous adaptations
    -----------------------------------------------
    Property               ViT          Swin         CLIP         DINOv2
    ---------              ---          ----         ----         ------
    Tensor format          (B,N,D)      (B,H,W,C)    (N,B,D)      (B,N,D)
    CLS index              [:,0,:]      global avg   [0,:,:]      [:,0,:]
    Attn return type       tuple(*)     Tensor       tuple        Tensor
    fix_encoder needed     YES(*)       NO           NO           NO
    patch_head tuple ret   YES(*)       NO           YES          NO
    Projection head        NO           NO           768->512     NO
    Loading API            torchvision  torchvision  clip.load()  torch.hub

    (*) The original ViT code needed fix_vit_encoder to patch block.forward
        because block.self_attention returned a tuple and the block tried to
        use it directly.  DINOv2's custom Attention.forward() returns a plain
        Tensor -- no patching of block.forward() is needed, and patch_head
        hooks can return a plain Tensor (unlike CLIP where the hook had to
        preserve the (tensor, weights) tuple for CLIP's internal [0] unpack).

    Token preparation
    -----------------
    DINOv2 exposes dino.prepare_tokens_with_masks() which does patch
    embedding + CLS prepend + positional encoding in one call, much
    cleaner than the manual assembly required for ViT and CLIP.

    Output dimension
    ----------------
    DINOv2 outputs embed_dim (768 for ViT-B) directly from the CLS token
    after the final LayerNorm -- no downstream projection like CLIP's
    768->512 visual.proj.  The new head is therefore Linear(768, num_classes).

    fp16 / dtype
    ------------
    torch.hub loads DINOv2 in fp32 by default -- no .float() conversion
    needed (contrast with clip.load() which defaults to fp16 on GPU).
    """
    def __init__(self, num_classes: int = NUM_CLASSES,
                 model_name: str = DINO_MODEL_NAME):
        super().__init__()

        # Downloads on first call; cached in torch.hub cache dir afterwards.
        # trust_repo=True suppresses the interactive prompt in PyTorch >= 1.13.
        self.dino = torch.hub.load(
            'facebookresearch/dinov2', model_name, trust_repo=True
        )

        # Freeze entire DINOv2 backbone
        for p in self.dino.parameters():
            p.requires_grad = False

        self.adapter_layers = DINO_ADAPTER_LAYERS
        hidden_dim = self.dino.embed_dim          # 768 for vitb14

        self.adapters = nn.ModuleDict({
            str(i): DINOAdapter(hidden_dim)
            for i in self.adapter_layers
        })

        # Classification head directly on DINOv2's 768-dim CLS embedding
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.dino    # shorthand

        # ----- Token preparation: patch embed + CLS prepend + pos encoding -----
        # Single call replaces the multi-step assembly needed for ViT and CLIP.
        x = d.prepare_tokens_with_masks(x)   # (B, N+1, D)   N = num_patches

        # ----- Transformer blocks with adapter injection -----
        for i, block in enumerate(d.blocks):
            x = block(x)                     # (B, N+1, D)
            if str(i) in self.adapters:
                x = self.adapters[str(i)](x)  # residual adapter

        # ----- Final LayerNorm + CLS classification -----
        x = d.norm(x)                        # (B, N+1, D)
        return self.head(x[:, 0])            # CLS token -> (B, num_classes)


model = DINOv2_Adapter(NUM_CLASSES).to(DEVICE)

# ------------------------------------------------------------------
# TRACER
# ------------------------------------------------------------------
def get_tensor(o):
    """
    Defensive helper: unwrap (tensor, weights) tuples if present.
    DINOv2's Attention.forward() returns a plain Tensor so this is a
    no-op -- kept for consistency with ViT / CLIP tracers.
    """
    return o[0] if isinstance(o, tuple) else o


def get_dino_attn_modules(model: DINOv2_Adapter) -> dict:
    """
    Collect all custom Attention modules from DINOv2's transformer blocks.
    Keys: "0", "1", ..., "11"  (flat indexing, same as ViT and CLIP).

    DINOv2 uses its own Attention class (NOT nn.MultiheadAttention).
    It returns a plain (B, N, D) Tensor -- no tuple, no fix_encoder.
    block.attn.num_heads gives the head count (12 for ViT-B/14).
    """
    return {str(i): block.attn for i, block in enumerate(model.dino.blocks)}


class CircuitTracer:
    def __init__(self, model: DINOv2_Adapter):
        self.model = model
        self.attn  = get_dino_attn_modules(model)

    # ------------------------------------------------------------------
    def run_with_cache(self, x: torch.Tensor):
        """
        Forward pass with hooks that store adapter and attention outputs.

        Cache tensor shapes (DINOv2):
          "mlp_{layer}"  -> (B, N, D)    adapter output
          "attn_{key}"   -> (B, N, D)    attention output

        Batch-first (B, N, D), identical to the original ViT code.
        get_tensor() is applied defensively but is a no-op for DINOv2.
        """
        cache: dict = {}
        hooks: list = []

        def save(name: str):
            def fn(m, inp, o):
                cache[name] = get_tensor(o).detach()
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
        Zero out channel `dim` of the CLS token in adapter `layer`.

        Identical to the original ViT code -- (B, N, D) format, CLS at [:, 0, :].

        ViT:  o[:, 0, dim] = 0   (B, N, D)
        Swin: o[:, :, :, dim]=0  (B, H, W, C) -- all spatial positions
        CLIP: o[0, :, dim] = 0   (N, B, D)    -- seq-first, CLS at dim-0
        DINO: o[:, 0, dim] = 0   (B, N, D)    -- same as ViT
        """
        def hook(m, inp, o):
            o = o.clone()
            o[:, 0, dim] = 0     # CLS token, all batches, specific channel
            return o

        h = self.model.adapters[layer].register_forward_hook(hook)
        out, cache = self.run_with_cache(x)
        h.remove()
        return out, cache

    # ------------------------------------------------------------------
    def patch_head(self, x: torch.Tensor, layer: str, head: int):
        """
        Zero out the contribution of attention head `head` in block `layer`.

        DINOv2's Attention returns a plain (B, N, D) Tensor.
        The hook returns a plain Tensor -- no tuple preservation needed.
        Logic is identical to the original ViT code.

        Contrast with CLIP where the hook had to return (tensor, weights)
        to survive CLIP's internal self.attn(...)[0] unpacking.

        ViT:  return plain tensor  (via get_tensor unwrap + reshape)
        CLIP: return (tensor, weights) tuple  <- CLIP-specific requirement
        DINO: return plain tensor  <- same as ViT, simpler than CLIP
        """
        attn_mod  = self.attn[layer]
        num_heads = attn_mod.num_heads    # 12 for vitb14

        def hook(m, inp, o):
            o = o.clone()
            B, N, D = o.shape
            Hd = D // num_heads
            o = o.view(B, N, num_heads, Hd)
            o[:, :, head, :] = 0
            return o.view(B, N, D)

        h = attn_mod.register_forward_hook(hook)
        out, cache = self.run_with_cache(x)
        h.remove()
        return out, cache

    # ------------------------------------------------------------------
    def compute(self, loader: DataLoader, max_batches: int = 1) -> dict:
        """
        Build per-class circuit graphs via activation patching.

        All indexing is identical to the original ViT code (B, N, D):
          MLP edge:  cache["mlp_"+l2][:, 0, d2].mean()
          Attn edge: cache["attn_"+l2][:, 0, :].mean()

        No axis-swap corrections needed (unlike CLIP's LND format).
        No spatial-mean corrections needed (unlike Swin's H,W dims).
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
                    D = cache["mlp_" + layer].shape[-1]    # 768

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
                layers_sorted  = sorted(self.attn.keys(), key=lambda k: int(k))
                adapter_sorted = sorted(self.model.adapters.keys(), key=int)

                # Attention -> Attention edges (consecutive blocks)
                for l1, l2 in zip(layers_sorted[:-1], layers_sorted[1:]):
                    for h1, _ in attn_top.get(l1, []):
                        _, patched_cache = self.patch_head(x, l1, h1)

                        for h2, _ in attn_top.get(l2, []):
                            # Identical to original ViT: [:, 0, :].mean()
                            eff = (
                                cache["attn_" + l2][:, 0, :].mean()
                                - patched_cache["attn_" + l2][:, 0, :].mean()
                            ).item()

                            if abs(eff) > 1e-4:
                                G.add_edge(
                                    f"{l1}_head_{h1}", f"{l2}_head_{h2}", weight=eff
                                )

                # Adapter -> Adapter edges (consecutive adapter stages)
                for l1, l2 in zip(adapter_sorted[:-1], adapter_sorted[1:]):
                    for d1, _ in mlp_top.get(l1, []):
                        _, patched_cache = self.patch_mlp(x, l1, d1)

                        for d2, _ in mlp_top.get(l2, []):
                            # Identical to original ViT: [:, 0, d2].mean()
                            eff = (
                                cache["mlp_" + l2][:, 0, d2].mean()
                                - patched_cache["mlp_" + l2][:, 0, d2].mean()
                            ).item()

                            if abs(eff) > 1e-4:
                                G.add_edge(
                                    f"{l1}_mlp_{d1}", f"{l2}_mlp_{d2}", weight=eff
                                )

            if i >= max_batches:
                break

        return circuits


# ------------------------------------------------------------------
# TRAINING  -- Sketch domain
# ------------------------------------------------------------------
dataset_sketch = datasets.ImageFolder(
    "/data/113-2/users/abanejee/mech_interp/pacs/pacs_data/pacs_data/sketch", transform=transform
)

generator_sketch  = torch.Generator().manual_seed(42)
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

model_sketch = DINOv2_Adapter(NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model_sketch.parameters()), lr=1e-3
)

best_val         = 0
patience_counter = 0


def train_epoch():
    model_sketch.train()
    total, correct = 0, 0
    for xb, yb in train_loader_sketch:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out  = model_sketch(xb)
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
        best_val         = val_acc
        patience_counter = 0
        torch.save(
            model_sketch.state_dict(),
            os.path.join(SAVE_DIR, "best_model_sketch_dino.pth")
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
    torch.load(os.path.join(SAVE_DIR, "best_model_sketch_dino.pth"))
)
model_sketch.eval()

# ------------------------------------------------------------------
# CIRCUIT EXTRACTION
# ------------------------------------------------------------------
print("\n=== CIRCUIT EXTRACTION ===")
tracer_sketch_dino   = CircuitTracer(model_sketch)
circuits_sketch_dino = tracer_sketch_dino.compute(train_loader_sketch)

# ------------------------------------------------------------------
# SAVE CIRCUITS
# ------------------------------------------------------------------
import pickle
circuit_path = os.path.join(SAVE_DIR, "circuits_dino_sketch.pkl")
with open(circuit_path, "wb") as f:
    pickle.dump(circuits_sketch_dino, f)

print(f"✅ Circuits saved at {circuit_path}")

# ------------------------------------------------------------------
# VISUALISATION
# ------------------------------------------------------------------
def visualize_layered(graph: nx.DiGraph, file: str = "circuit_layered_dino.html"):
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
                title=f"Block {layer}",
            )

    for u, v, data in graph.edges(data=True):
        net.add_edge(u, v, value=abs(data.get("weight", 1)))

    net.write_html(file)
    print(f"Saved visualization to {file}")


cls = list(circuits_sketch_dino.keys())[0]
print(cls)
print(circuits_sketch_dino[cls])
# visualize_layered(circuits_sketch_dino[cls])

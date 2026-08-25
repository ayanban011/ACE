import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import clip   # pip install git+https://github.com/openai/CLIP.git
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
NUM_WORKERS     = 1
EPOCHS          = 10000
TOPK            = 5
PATIENCE        = 30
SAVE_DIR        = "./outputs_pacs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------------------------
# CLIP architecture constants  (ViT-B/16 defaults)
# ------------------------------------------------------------------
CLIP_MODEL_NAME     = "ViT-B/16"
#
# ViT-B/16 internal layout:
#   visual.transformer.resblocks  -> 12 ResidualAttentionBlocks (idx 0-11)
#   each block: block.attn        -> nn.Multihead>Attention (12 heads, dim=768)
#   after transformer: visual.proj (768 -> 512) maps CLS to the shared CLIP space
#
CLIP_HIDDEN_DIM     = 768   # transformer internal width  (visual.width)
CLIP_PROJ_DIM       = 512   # visual.proj output dimension
CLIP_ADAPTER_LAYERS = [8, 10, 11]   # mirror original ViT code (last 3 blocks)
#
# To switch to ViT-L/14 instead:
#   CLIP_MODEL_NAME = "ViT-L/14"
#   CLIP_HIDDEN_DIM = 1024
#   CLIP_PROJ_DIM   = 768

# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
# CLIP has its own normalisation statistics (different from ImageNet).
# We replicate clip.load()'s preprocess but add RandomHorizontalFlip
# for training augmentation.
transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std= [0.26862954, 0.26130258, 0.27577711],
    ),
])

# ------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------
class CLIPAdapter(nn.Module):
    """
    Bottleneck residual adapter inserted into a CLIP transformer block.

    Input/output shape: (N, B, D) -- CLIP's internal LND (seq-first) format.

    Identical algebra to ViTAdapter / SwinAdapter; nn.Linear acts on the
    last dim D regardless of leading dimensions, so no reshape is needed.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, B, D)  -- seq-first (LND)
        return x + self.mlp(x)


class CLIP_Adapter(nn.Module):
    """
    CLIP ViT backbone (frozen) + lightweight adapters + new classifier head.

    Key differences from ViT_Adapter and Swin_Adapter
    --------------------------------------------------
    1. Tensor format inside the transformer is (N, B, D) -- LND / seq-first.
       ViT used (B, N, D) and Swin used (B, H, W, C).
       => CLS token lives at index [0, :, :] (dim 0) instead of [:, 0, :] (dim 1).

    2. block.attn is nn.MultiheadAttention, which returns a (tensor, weights)
       tuple -- same root issue as ViT's self_attention.
       CLIP's ResidualAttentionBlock already unwraps [0] internally via its
       attention() helper, so block.forward() is correct without patching.
       HOWEVER, our patch_head() hooks block.attn directly; it MUST return a
       (tensor, weights) tuple so CLIP's [0] unpacking gets the tensor, not
       the first row of the tensor.
       => No fix_encoder() patch needed; only the hook return type changes.

    3. After the transformer the CLS token is projected from CLIP_HIDDEN_DIM
       (768) to CLIP_PROJ_DIM (512) via visual.proj (a frozen Parameter).
       Our new classification head sits on top of that 512-dim CLIP space.

    4. clip.load() produces fp16 weights on GPU; we load on CPU (fp32) then
       move to DEVICE, keeping adapter gradients in fp32 throughout.
    """
    def __init__(self, num_classes: int = NUM_CLASSES,
                 model_name: str = CLIP_MODEL_NAME):
        super().__init__()

        # Load on CPU first to guarantee fp32 (CLIP defaults to fp16 on GPU)
        clip_model, _ = clip.load(model_name, device="cpu")
        self.clip = clip_model.float()      # ensure fp32

        # Freeze entire CLIP backbone (visual + text)
        for p in self.clip.parameters():
            p.requires_grad = False

        self.adapter_layers = CLIP_ADAPTER_LAYERS
        hidden_dim = getattr(self.clip.visual, 'width', None)
        if hidden_dim is None:
            hidden_dim = getattr(self.clip.visual, 'embed_dim', self.clip.visual.conv1.weight.shape[0])

        self.adapters = nn.ModuleDict({
            str(i): CLIPAdapter(hidden_dim)
            for i in self.adapter_layers
        })

        # Head lives on top of the CLIP projected space (512-dim for ViT-B/16)
        proj_dim = self.clip.visual.proj.shape[1]           # 512 for ViT-B/16
        self.head = nn.Linear(proj_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.clip.visual                                # shorthand

        # ----- Patch embedding (replicates VisionTransformer.forward) -----
        x = v.conv1(x)                                     # (B, D, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1)         # (B, D, N_patches)
        x = x.permute(0, 2, 1)                             # (B, N_patches, D)

        # Prepend CLS token
        cls = v.class_embedding.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        cls = cls.expand(x.shape[0], -1, -1)               # (B, 1, D)
        x   = torch.cat([cls, x], dim=1)                   # (B, N+1, D)

        x = x + v.positional_embedding                     # (B, N+1, D)
        x = v.ln_pre(x)

        # CLIP's transformer expects (N, B, D) -- LND / seq-first
        x = x.permute(1, 0, 2)                            # (N+1, B, D)

        # ----- Transformer blocks with adapter injection -----
        for i, block in enumerate(v.transformer.resblocks):
            x = block(x)                                   # (N+1, B, D)
            if str(i) in self.adapters:
                x = self.adapters[str(i)](x)               # residual adapter

        # ----- Head pipeline -----
        x = x.permute(1, 0, 2)                            # (B, N+1, D)
        x = v.ln_post(x[:, 0, :])                         # CLS token -> (B, D)

        if v.proj is not None:
            x = x @ v.proj                                 # (B, proj_dim=512)

        return self.head(x)                                # (B, num_classes)


model = CLIP_Adapter(NUM_CLASSES).to(DEVICE)

# ------------------------------------------------------------------
# TRACER
# ------------------------------------------------------------------
def get_tensor(o):
    """
    Unwrap (tensor, attn_weights) tuples from nn.MultiheadAttention.
    Identical helper to the original ViT code; still needed for CLIP.
    """
    return o[0] if isinstance(o, tuple) else o


def get_clip_attn_modules(model: CLIP_Adapter) -> dict:
    """
    Collect all nn.MultiheadAttention modules from CLIP's visual transformer.
    Keys: "0", "1", ..., "11"  (same flat indexing as the original ViT code).
    """
    return {
        str(i): block.attn
        for i, block in enumerate(model.clip.visual.transformer.resblocks)
    }


class CircuitTracer:
    def __init__(self, model: CLIP_Adapter):
        self.model = model
        self.attn  = get_clip_attn_modules(model)

    # ------------------------------------------------------------------
    def run_with_cache(self, x: torch.Tensor):
        """
        Full forward pass with hooks that store adapter and attention outputs.

        Cache tensor shapes (CLIP):
          "mlp_{layer}"  -> (N, B, D)    adapter output  [LND]
          "attn_{key}"   -> (N, B, D)    attn output after get_tensor() [LND]

        Comparison:
          ViT   stored (B, N, D), CLS at [:, 0, :]
          Swin  stored (B, H, W, C), global avg used
          CLIP  stores (N, B, D), CLS at [0, :, >:]
        """
        cache: dict = {}
        hooks: list = []

        def save(name: str):
            def fn(m, inp, o):
                # get_tensor() unwraps the (tensor, None) tuple from block.attn
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

        Tensor format comparison:
          ViT:  (B, N, D) -> zero o[:, 0, dim] >   (CLS is seq position 0, batch dim 0)
          Swin: (B,H,W,C) -> zero o[:, :, :, dim] (all spatial positions)
          CLIP: (N, B, D) -> zero o[0, :, dim]    (CLS is seq position 0, now in dim 0)
        """
        def hook(m, inp, o):
            o = o.clone()
            o[0, :, dim] = 0    # seq pos 0 = CLS token, all batches, specific channel
            return o

        h = self.model.adapters[layer].register_forward_hook(hook)
        out, cache = self.run_with_cache(x)
        h.remove()
        return out, cache

    # ------------------------------------------------------------------
    def patch_head(self, x: torch.Tensor, layer: str, head: int):
        """
        Zero out the contribution of attention head `head` in block `layer`.

        CRITICAL CLIP-SPECIFIC REQUIREMENT:
        block.attn (nn.MultiheadAttention) returns (tensor, weights).
        CLIP's ResidualAttentionBlock.attention() unpacks it with [0]:
            return self.attn(x, x, x, need_weights=False, ...)[0]
        If our hook returns a plain Tensor, CLIP silently does tensor[0],
        slicing the first row instead of unpacking the tuple.
        The hook MUST return a (tensor, weights) tuple.

        Tensor format comparison:
          ViT:  (B, N, D) -> reshape (B,N,H,Hd); zero head h; reshape; return tensor
          Swin: (B,H,W,C) -> zero slice along C;             return tensor
          CLIP: (N, B, D) -> zero slice along D;             return (tensor, weights)
        """
        attn_mod  = self.attn[layer]
        num_heads = attn_mod.num_heads              # 12 for ViT-B/16

        def hook(m, inp, o):
            attn_out, attn_weights = o              # (N, B, D), None
            attn_out = attn_out.clone()
            N, B, D  = attn_out.shape
            Hd       = D // num_heads
            attn_out[:, :, head * Hd:(head + 1) * Hd] = 0
            return (attn_out, attn_weights)         # <- preserve tuple for CLIP's [0] unpack

        h = attn_mod.register_forward_hook(hook)
        out, cache = self.run_with_cache(x)
        h.remove()
        return out, cache

    # ------------------------------------------------------------------
    def compute(self, loader: DataLoader, max_batches: int = 1) -> dict:
        """
        Build per-class circuit graphs via activation patching.

        CLIP-specific changes vs ViT version:
          MLP edge:  cache["mlp_"+l2][:, 0, d2].mean()
                  -> cache["mlp_"+l2][0, :, d2].mean()   (LND axis swap)

          Attn edge: cache["attn_"+l2][:, 0, :].mean()
                  -> cache["attn_"+l2][0, :, :].mean()   (LND axis swap)
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
                    D = cache["mlp_" + layer].shape[-1]    # hidden_dim (768)

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
                            # CLIP: CLS at seq-dim 0 -> [0, :, :].mean()
                            # ViT equivalent was:      [:, 0, :].mean()
                            eff = (
                                cache["attn_" + l2][0, :, :].mean()
                                - patched_cache["attn_" + l2][0, :, :].mean()
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
                            # CLIP: [0, :, d2] -> CLS across all batch items
                            # ViT equivalent was: [:, 0, d2].mean()
                            eff = (
                                cache["mlp_" + l2][0, :, d2].mean()
                                - patched_cache["mlp_" + l2][0, :, d2].mean()
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
    "./pacs/pacs_data/pacs_data/sketch", transform=transform
)

generator_sketch  = torch.Generator().manual_seed(42)
train_size_sketch = int(0.8 * len(dataset_sketch))
test_size_sketch  = len(dataset_sketch) - train_size_sketch

train_dataset_sketch, test_dataset_sketch = random_split(
    dataset_sketch, [train_size_sketch, test_size_sketch], generator=generator_sketch
)

train_loader_sketch = DataLoader(
    train_dataset_sketch, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False
)
test_loader_sketch = DataLoader(
    test_dataset_sketch, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=False
)

model_sketch = CLIP_Adapter(NUM_CLASSES).to(DEVICE)

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
            os.path.join(SAVE_DIR, "best_model_sketch_clip.pth")
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
    torch.load(os.path.join(SAVE_DIR, "best_model_sketch_clip.pth"))
)
model_sketch.eval()

# ------------------------------------------------------------------
# CIRCUIT EXTRACTION
# ------------------------------------------------------------------
print("\n=== CIRCUIT EXTRACTION ===")
tracer_sketch_clip   = CircuitTracer(model_sketch)
circuits_sketch_clip = tracer_sketch_clip.compute(train_loader_sketch)

# ------------------------------------------------------------------
# SAVE CIRCUITS
# ------------------------------------------------------------------
import pickle
circuit_path = os.path.join(SAVE_DIR, "circuits_clip_sketch.pkl")
with open(circuit_path, "wb") as f:
    pickle.dump(circuits_sketch_clip, f)

print(f"✅ Circuits saved at {circuit_path}")

# ------------------------------------------------------------------
# VISUALISATION
# ------------------------------------------------------------------
def visualize_layered(graph: nx.DiGraph, file: str = "circuit_layered_clip.html"):
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


cls = list(circuits_sketch_clip.keys())[0]
print(cls)
print(circuits_sketch_clip[cls])
# visualize_layered(circuits_sketch_clip[cls])

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Set torch cache directory

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
TOPK = 5   # important neurons per layer
PATIENCE = 30
SAVE_DIR = "./outputs_domainnet"
os.makedirs(SAVE_DIR, exist_ok=True)


# -----------------------------
# DATA
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# dataset = datasets.ImageFolder("/home/aban/circuit_tracing/DomainNet/clipart", transform=transform)

# generator = torch.Generator().manual_seed(42)
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size

# train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers = 4)

# -----------------------------
# MODEL
# -----------------------------
class ViTAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim//4),
            nn.ReLU(),
            nn.Linear(dim//4, dim)
        )

    def forward(self, x):
        return x + self.mlp(x)

class ViT_Adapter(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.vit = models.vit_b_16(weights="IMAGENET1K_V1")

        for p in self.vit.parameters():
            p.requires_grad = False

        self.adapter_layers = [8,10,11]

        self.adapters = nn.ModuleDict({
            str(i): ViTAdapter(self.vit.hidden_dim)
            for i in self.adapter_layers
        })

        self.head = nn.Linear(self.vit.hidden_dim, num_classes)

    def forward(self, x):
        x = self.vit._process_input(x)
        B = x.shape[0]

        cls_token = self.vit.class_token.expand(B,-1,-1)
        x = torch.cat([cls_token,x],dim=1)

        for i, block in enumerate(self.vit.encoder.layers):
            x = block(x)
            if str(i) in self.adapters:
                x = self.adapters[str(i)](x)

        x = self.vit.encoder.ln(x)
        return self.head(x[:,0])

# -----------------------------
# 🔥 FIX VIT BUG (CRITICAL)
# -----------------------------
def fix_vit_encoder(model):
    for block in model.vit.encoder.layers:

        def new_forward(self, input):
            x = self.ln_1(input)

            attn_out = self.self_attention(x, x, x, need_weights=False)

            if isinstance(attn_out, tuple):
                x = attn_out[0]
            else:
                x = attn_out

            x = self.dropout(x)
            x = x + input

            y = self.ln_2(x)
            y = self.mlp(y)

            return x + y

        block.forward = new_forward.__get__(block, block.__class__)

model = ViT_Adapter(NUM_CLASSES).to(DEVICE)
fix_vit_encoder(model)

# -----------------------------
# TRAINING
# -----------------------------
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=1e-3)

# best_val = 0
# patience_counter = 0

# def train_epoch():
#     model.train()
#     total, correct = 0,0
#     for x,y in train_loader:
#         x,y = x.to(DEVICE), y.to(DEVICE)

#         optimizer.zero_grad()
#         out = model(x)
#         loss = criterion(out,y)
#         loss.backward()
#         optimizer.step()

#         correct += (out.argmax(1)==y).sum().item()
#         total += y.size(0)
#     return correct/total

# @torch.no_grad()
# def evaluate():
#     model.eval()
#     total, correct = 0,0
#     for x,y in test_loader:
#         x,y = x.to(DEVICE), y.to(DEVICE)
#         out = model(x)
#         correct += (out.argmax(1)==y).sum().item()
#         total += y.size(0)
#     return correct/total

# print("=== TRAINING ===")
# for e in range(EPOCHS):
#     tr = train_epoch()
#     val = evaluate()
#     print(f"Epoch {e+1}: {tr:.3f} | {val:.3f}")

#     if val > best_val:
#         best_val = val
#         patience_counter = 0
#         torch.save(model.state_dict(), f"{SAVE_DIR}/best_model_clipart_vit.pth")
#     else:
#         patience_counter += 1

#     if patience_counter >= PATIENCE:
#         print("Early stopping")
#         break

# # -----------------------------
# # LOAD BEST MODEL
# # -----------------------------
# model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model_clipart_vit.pth")))
# model.eval()

# -----------------------------
# TRACER
# -----------------------------
def get_tensor(o):
    return o[0] if isinstance(o, tuple) else o
# -----------------------------
# TRACER
# -----------------------------
def get_attn_modules(model):
    return {str(i): block.self_attention for i, block in enumerate(model.vit.encoder.layers)}

class CircuitTracer:
    def __init__(self, model):
        self.model = model
        self.attn = get_attn_modules(model)

    def run_with_cache(self, x):
        cache = {}
        hooks = []

        def save(name):
            def fn(m,i,o):
                cache[name] = get_tensor(o).detach()
            return fn

        for n,m in self.model.adapters.items():
            hooks.append(m.register_forward_hook(save("mlp_"+n)))

        for n,m in self.attn.items():
            hooks.append(m.register_forward_hook(save("attn_"+n)))

        out = self.model(x)

        for h in hooks: h.remove()
        return out, cache

    def patch_mlp(self, x, layer, dim):
        def hook(m,i,o):
            o = get_tensor(o).clone()
            o[:,0,dim] = 0
            return o

        h = self.model.adapters[layer].register_forward_hook(hook)
        out, cache = self.run_with_cache(x)
        h.remove()
        return out, cache

    def patch_head(self, x, layer, head):
        attn = self.attn[layer]
        H = attn.num_heads

        def hook(m,i,o):
            o = get_tensor(o)
            B,N,D = o.shape
            Hd = D//H
            o = o.clone().view(B,N,H,Hd)
            o[:,:,head] = 0
            return o.view(B,N,D)

        h = attn.register_forward_hook(hook)
        out, cache = self.run_with_cache(x)
        h.remove()
        return out, cache

    def compute(self, loader, max_batches=1):
        circuits = {}

        for i,(x,y) in enumerate(loader):
            x,y = x.to(DEVICE), y.to(DEVICE)
            out, cache = self.run_with_cache(x)

            for cls in y.unique():
                cls = cls.item()
                idx = (y==cls)
                base = out[idx,cls].mean()

                if cls not in circuits:
                    circuits[cls] = nx.DiGraph()

                G = circuits[cls]

                mlp_top, attn_top = {}, {}

                # ---- MLP nodes ----
                for layer in self.model.adapters:
                    D = cache["mlp_"+layer].shape[-1]

                    scores = []
                    for d in torch.randperm(D)[:TOPK]:
                        patched,_ = self.patch_mlp(x, layer, d.item())
                        eff = (base - patched[idx,cls].mean()).item()
                        scores.append((d.item(), eff))

                    scores.sort(key=lambda z: abs(z[1]), reverse=True)
                    mlp_top[layer] = scores[:TOPK]

                    for d,eff in mlp_top[layer]:
                        G.add_node(f"{layer}_mlp_{d}", layer=layer, importance=eff)

                # ---- Attention nodes ----
                for layer in self.attn:
                    H = self.attn[layer].num_heads

                    scores = []
                    for h in range(H):
                        patched,_ = self.patch_head(x, layer, h)
                        eff = (base - patched[idx,cls].mean()).item()
                        scores.append((h, eff))

                    scores.sort(key=lambda z: abs(z[1]), reverse=True)
                    attn_top[layer] = scores[:TOPK]

                    for h,eff in attn_top[layer]:
                        G.add_node(f"{layer}_head_{h}", layer=layer, importance=eff)

                # ---- EDGES ----
                layers = sorted(self.attn.keys(), key=lambda x:int(x))

                for l1, l2 in zip(layers[:-1], layers[1:]):

                    for h1,_ in attn_top.get(l1,[]):
                        patched_out, patched_cache = self.patch_head(x, l1, h1)

                        for h2,_ in attn_top.get(l2,[]):
                            eff = (cache["attn_"+l2][:,0,:].mean() -
                                   patched_cache["attn_"+l2][:,0,:].mean()).item()

                            if abs(eff) > 1e-4:
                                G.add_edge(f"{l1}_head_{h1}", f"{l2}_head_{h2}", weight=eff)

                    for d1,_ in mlp_top.get(l1,[]):
                        patched_out, patched_cache = self.patch_mlp(x, l1, d1)

                        for d2,_ in mlp_top.get(l2,[]):
                            eff = (cache["mlp_"+l2][:,0,d2].mean() -
                                   patched_cache["mlp_"+l2][:,0,d2].mean()).item()

                            if abs(eff) > 1e-4:
                                G.add_edge(f"{l1}_mlp_{d1}", f"{l2}_mlp_{d2}", weight=eff)

            if i >= max_batches:
                break

        return circuits

# # -----------------------------
# # EXTRACT CIRCUITS
# # -----------------------------
# # print("=== EXTRACTING CIRCUITS ===")
# # tracer = CircuitTracer(model)
# # circuits = tracer.compute(train_loader)

# # -----------------------------
# # SAVE CIRCUITS
# # -----------------------------
# import pickle
# circuit_path = os.path.join(SAVE_DIR, "circuits_vit_clipart.pkl")
# with open(circuit_path, "wb") as f:
#     pickle.dump(circuits, f)

# print(f"✅ Circuits saved at {circuit_path}")

# def visualize_layered(graph, file="circuit_layered.html"):
#     net = Network(directed=True)

#     layer_groups = {}
#     for node, data in graph.nodes(data=True):
#         layer = data.get("layer", "unknown")
#         layer_groups.setdefault(layer, []).append(node)

#     x_gap, y_gap = 200, 50

#     for i, (layer, nodes) in enumerate(sorted(layer_groups.items())):
#         for j, node in enumerate(nodes):
#             net.add_node(
#                 node,
#                 x=i*x_gap,
#                 y=j*y_gap,
#                 fixed=True,
#                 value=abs(graph.nodes[node].get("importance", 1)),
#                 title=f"Layer {layer}"
#             )

#     for u, v, data in graph.edges(data=True):
#         net.add_edge(u, v, value=abs(data.get("weight", 1)))

#     net.write_html(file)   # ✅ FIX
#     print(f"Saved visualization to {file}")

# # visualize one class
# cls = list(circuits.keys())[0]
# print(cls)
# print(circuits[cls])
# visualize_layered(circuits[cls])


# dataset_cartoon = datasets.ImageFolder("/home/aban/circuit_tracing/DomainNet/infograph", transform=transform)

# generator_cartoon = torch.Generator().manual_seed(42)
# train_size_cartoon = int(0.8 * len(dataset_cartoon))
# test_size_cartoon = len(dataset_cartoon) - train_size_cartoon

# train_dataset_cartoon, test_dataset_cartoon = random_split(dataset_cartoon, [train_size_cartoon, test_size_cartoon], generator=generator_cartoon)

# train_loader_cartoon = DataLoader(train_dataset_cartoon, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
# test_loader_cartoon = DataLoader(test_dataset_cartoon, batch_size=BATCH_SIZE, num_workers = 4)

# model_cartoon = ViT_Adapter(NUM_CLASSES).to(DEVICE)
# fix_vit_encoder(model_cartoon)

# # -----------------------------
# # TRAIN WITH EARLY STOPPING
# # -----------------------------
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_cartoon.parameters()), lr=1e-3)

# best_val = 0
# patience_counter = 0


# def train_epoch():
#     model_cartoon.train()
#     total, correct = 0,0
#     for x,y in train_loader_cartoon:
#         x,y = x.to(DEVICE), y.to(DEVICE)
#         optimizer.zero_grad()
#         out = model_cartoon(x)
#         loss = criterion(out,y)
#         loss.backward()
#         optimizer.step()

#         pred = out.argmax(1)
#         correct += (pred==y).sum().item()
#         total += y.size(0)

#     return correct/total


# @torch.no_grad()
# def evaluate():
#     model_cartoon.eval()
#     total, correct = 0,0
#     for x,y in test_loader_cartoon:
#         x,y = x.to(DEVICE), y.to(DEVICE)
#         out = model_cartoon(x)
#         pred = out.argmax(1)
#         correct += (pred==y).sum().item()
#         total += y.size(0)
#     return correct/total


# print("=== TRAINING ===")
# for e in range(EPOCHS):
#     train_acc = train_epoch()
#     val_acc = evaluate()

#     print(f"Epoch {e+1}: Train {train_acc:.3f} | Val {val_acc:.3f}")

#     # Early stopping logic
#     if val_acc > best_val:
#         best_val = val_acc
#         patience_counter = 0
#         torch.save(model_cartoon.state_dict(), os.path.join(SAVE_DIR, "best_model_infograph_vit.pth"))
#         print("✅ Model saved")
#     else:
#         break

# # -----------------------------
# # LOAD BEST MODEL
# # -----------------------------
# model_cartoon.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model_infograph_vit.pth")))
# model_cartoon.eval()

# print("\n=== CIRCUIT EXTRACTION ===")
# tracer_cartoon_vit = CircuitTracer(model_cartoon)
# circuits_cartoon_vit = tracer_cartoon_vit.compute(train_loader_cartoon)

# # -----------------------------
# # SAVE CIRCUITS
# # -----------------------------
# import pickle
# circuit_path = os.path.join(SAVE_DIR, "circuits_vit_infograph.pkl")
# with open(circuit_path, "wb") as f:
#     pickle.dump(circuits_cartoon_vit, f)

# print(f"✅ Circuits saved at {circuit_path}")


# # visualize one class
# cls = list(circuits_cartoon_vit.keys())[0]
# print(cls)
# print(circuits_cartoon_vit[cls])
# visualize_layered(circuits_cartoon_vit[cls])

# @torch.no_grad()
# def circuit_faithfulness(model, circuits, loader, device):
#     model.eval()

#     results = {}

#     for cls, graph in circuits.items():
#         print(f"Evaluating class {cls}")

#         total, correct_full, correct_pruned = 0, 0, 0

#         for x, y in loader:
#             x, y = x.to(device), y.to(device)

#             # full prediction
#             out_full = model(x)
#             pred_full = out_full.argmax(1)

#             # prune circuit neurons
#             def prune_hook(layer_name, channels):
#                 def hook(m, i, o):
#                     o = o.clone()
#                     for ch in channels:
#                         o[:, ch] = 0
#                     return o
#                 return hook

#             hooks = []
#             layer_channels = {}

#             # group nodes by layer
#             for node in graph.nodes():
#                 layer, ch = node.split("_ch")
#                 layer_channels.setdefault(layer, []).append(int(ch))

#             for layer, chs in layer_channels.items():
#                 handle = model.adapters[layer].register_forward_hook(
#                     prune_hook(layer, chs)
#                 )
#                 hooks.append(handle)

#             out_pruned = model(x)
#             pred_pruned = out_pruned.argmax(1)

#             for h in hooks:
#                 h.remove()

#             correct_full += (pred_full == y).sum().item()
#             correct_pruned += (pred_pruned == y).sum().item()
#             total += y.size(0)

#         acc_full = correct_full / total
#         acc_pruned = correct_pruned / total

#         results[cls] = {
#             "full_acc": acc_full,
#             "pruned_acc": acc_pruned,
#             "drop": acc_full - acc_pruned
#         }

#         print(f"Drop: {results[cls]['drop']:.4f}")

#     return results

# import networkx as nx

# def extract_top_paths(graph, top_k=5):
#     paths = []

#     # ensure DAG
#     if not nx.is_directed_acyclic_graph(graph):
#         graph = nx.DiGraph([(u,v,d) for u,v,d in graph.edges(data=True)])

#     for source in graph.nodes():
#         for target in graph.nodes():
#             if source == target:
#                 continue

#             try:
#                 path = nx.shortest_path(graph, source, target)
#                 weight = 0

#                 for u, v in zip(path[:-1], path[1:]):
#                     weight += abs(graph[u][v].get("weight", 0))

#                 paths.append((path, weight))

#             except:
#                 continue

#     paths = sorted(paths, key=lambda x: x[1], reverse=True)[:top_k]
#     return paths

# paths = extract_top_paths(circuits[0], top_k=10)

# for p, w in paths:
#     print(" -> ".join(p), "| score:", w)

# def compare_circuits(circuits_a, circuits_b):
#     results = {}

#     for cls in circuits_a:
#         nodes_a = set(circuits_a[cls].nodes())
#         nodes_b = set(circuits_b[cls].nodes())

#         intersection = nodes_a & nodes_b
#         union = nodes_a | nodes_b

#         jaccard = len(intersection) / len(union) if len(union) > 0 else 0

#         results[cls] = {
#             "overlap": len(intersection),
#             "jaccard": jaccard
#         }

#     return results

# results = compare_circuits(circuits, circuits_cartoon_vit)

# for cls, stats in results.items():
#     print(f"Class {cls}: Jaccard {stats['jaccard']:.3f}")

dataset_photo = datasets.ImageFolder("/home/aban/circuit_tracing/DomainNet/sketch", transform=transform)

generator_photo = torch.Generator().manual_seed(42)
train_size_photo = int(0.8 * len(dataset_photo))
test_size_photo = len(dataset_photo) - train_size_photo

train_dataset_photo, test_dataset_photo = random_split(dataset_photo, [train_size_photo, test_size_photo], generator=generator_photo)

train_loader_photo = DataLoader(train_dataset_photo, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
test_loader_photo = DataLoader(test_dataset_photo, batch_size=BATCH_SIZE, num_workers = 4)

model_photo = ViT_Adapter(NUM_CLASSES).to(DEVICE)
fix_vit_encoder(model_photo)

# -----------------------------
# TRAIN WITH EARLY STOPPING
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_photo.parameters()), lr=1e-3)

best_val = 0
patience_counter = 0


def train_epoch():
    model_photo.train()
    total, correct = 0,0
    for x,y in train_loader_photo:
        x,y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model_photo(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(1)
        correct += (pred==y).sum().item()
        total += y.size(0)

    return correct/total


@torch.no_grad()
def evaluate():
    model_photo.eval()
    total, correct = 0,0
    for x,y in test_loader_photo:
        x,y = x.to(DEVICE), y.to(DEVICE)
        out = model_photo(x)
        pred = out.argmax(1)
        correct += (pred==y).sum().item()
        total += y.size(0)
    return correct/total


print("=== TRAINING ===")
for e in range(EPOCHS):
    train_acc = train_epoch()
    val_acc = evaluate()

    print(f"Epoch {e+1}: Train {train_acc:.3f} | Val {val_acc:.3f}")

    # Early stopping logic
    if val_acc > best_val:
        best_val = val_acc
        patience_counter = 0
        torch.save(model_photo.state_dict(), os.path.join(SAVE_DIR, "best_model_sketch _vit.pth"))
        print("✅ Model saved")
    else:
        break

# -----------------------------
# LOAD BEST MODEL
# -----------------------------
model_photo.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model_sketch _vit.pth")))
model_photo.eval()

print("\n=== CIRCUIT EXTRACTION ===")
tracer_photo_vit = CircuitTracer(model_photo)
circuits_photo_vit = tracer_photo_vit.compute(train_loader_photo)

# -----------------------------
# SAVE CIRCUITS
# -----------------------------
import pickle
circuit_path = os.path.join(SAVE_DIR, "circuits_vit_sketch.pkl")
with open(circuit_path, "wb") as f:
    pickle.dump(circuits_photo_vit, f)

print(f"✅ Circuits saved at {circuit_path}")


# visualize one class
cls = list(circuits_photo_vit.keys())[0]
print(cls)
print(circuits_photo_vit[cls])
#visualize_layered(circuits_photo_vit[cls])

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Set torch cache directory

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
BATCH_SIZE = 32
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

dataset_artpainting = datasets.ImageFolder("/home/aban/circuit_tracing/DomainNet/clipart", transform=transform)

generator = torch.Generator().manual_seed(42)
train_size = int(0.8 * len(dataset_artpainting))
test_size = len(dataset_artpainting) - train_size

train_dataset_artpainting, test_dataset_artpainting = random_split(dataset_artpainting, [train_size, test_size], generator=generator)

train_loader_artpainting = DataLoader(train_dataset_artpainting, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
test_loader_artpainting = DataLoader(test_dataset_artpainting, batch_size=BATCH_SIZE, num_workers = 4)

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

# -----------------------------
# MODEL (ResNet50 version)
# -----------------------------
class ResNet50_Adapter(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        base = models.resnet50(weights="IMAGENET1K_V2")

        # Freeze backbone
        for p in base.parameters():
            p.requires_grad = False

        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        # ResNet stages
        self.layer1 = base.layer1   # 256
        self.layer2 = base.layer2   # 512
        self.layer3 = base.layer3   # 1024
        self.layer4 = base.layer4   # 2048

        # Insert adapters AFTER each residual stage
        self.adapters = nn.ModuleDict({
            "layer1": MLPAdapter(256),
            "layer2": MLPAdapter(512),
            "layer3": MLPAdapter(1024),
            "layer4": MLPAdapter(2048),
        })

        self.pool = base.avgpool

        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.adapters["layer1"](x)

        x = self.layer2(x)
        x = self.adapters["layer2"](x)

        x = self.layer3(x)
        x = self.adapters["layer3"](x)

        x = self.layer4(x)
        x = self.adapters["layer4"](x)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        return self.classifier(x)

model_resnet50_artpainting = ResNet50_Adapter(NUM_CLASSES).to(DEVICE)

# -----------------------------
# TRAIN WITH EARLY STOPPING
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_resnet50_artpainting.parameters()), lr=1e-3)

best_val = 0
patience_counter = 0


def train_epoch():
    model_resnet50_artpainting.train()
    total, correct = 0,0
    for x,y in train_loader_artpainting:
        x,y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model_resnet50_artpainting(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(1)
        correct += (pred==y).sum().item()
        total += y.size(0)

    return correct/total


@torch.no_grad()
def evaluate():
    model_resnet50_artpainting.eval()
    total, correct = 0,0
    for x,y in test_loader_artpainting:
        x,y = x.to(DEVICE), y.to(DEVICE)
        out = model_resnet50_artpainting(x)
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
        torch.save(model_resnet50_artpainting.state_dict(), os.path.join(SAVE_DIR, "best_model_clipart_resnet50.pth"))
        print("✅ Model saved")
    else:
        break

# -----------------------------
# LOAD BEST MODEL
# -----------------------------
model_resnet50_artpainting.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model_clipart_resnet50.pth")))
model_resnet50_artpainting.eval()

# -----------------------------
# CIRCUIT TRACER
# -----------------------------
class AdapterCircuitTracer:
    def __init__(self, model):
        self.model = model

    def run_with_cache(self, x):
        cache = {}
        hooks = []

        def hook_fn(name):
            def fn(m,i,o): cache[name] = o.detach()
            return fn

        for name, module in self.model.adapters.items():
            hooks.append(module.register_forward_hook(hook_fn(name)))

        out = self.model(x)

        for h in hooks: h.remove()
        return out, cache

    #def patch(self, x, layer_name, ch):
    #    def hook(m,i,o):
    #        o = o.clone()
    #        o[:, ch] = 0
    #        return o

    #    handle = self.model.adapters[layer_name].register_forward_hook(hook)
    #    out = self.model(x)
    #    handle.remove()
    #    return out
    def patch_and_run(self, x, layer_name, ch):
        cache = {}
        hooks = []

        def patch_hook(m, i, o):
            o = o.clone()
            o[:, ch] = 0
            return o

        def save_hook(name):
            def fn(m, i, o):
                cache[name] = o.detach()
            return fn

        # register patch hook
        patch_handle = self.model.adapters[layer_name].register_forward_hook(patch_hook)

        # register cache hooks
        for name, module in self.model.adapters.items():
            hooks.append(module.register_forward_hook(save_hook(name)))

        out = self.model(x)

        patch_handle.remove()
        for h in hooks:
            h.remove()

        return out, cache
    
    

    def compute_circuits(self, loader, max_batches=3):
        circuits = {}

        for i,(x,y) in enumerate(loader):
            x,y = x.to(DEVICE), y.to(DEVICE)
            out, cache = self.run_with_cache(x)

            for cls in y.unique():
                cls = cls.item()
                idx = (y==cls)
                base = out[idx, cls].mean()

                if cls not in circuits:
                    circuits[cls] = nx.DiGraph()

                layer_topk = {}

                for layer in cache:
                    act = cache[layer]
                    C = act.shape[1]
                    scores = []

                    for c in range(C):
                        #patched = self.patch_and_run(x, layer, c)
                        #eff = (base - patched[idx, cls].mean()).item()
                        patched_out, _ = self.patch_and_run(x, layer, c)
                        eff = (base - patched_out[idx, cls].mean()).item()
                        scores.append((c, eff))

                    scores = sorted(scores, key=lambda x: abs(x[1]), reverse=True)[:TOPK]
                    layer_topk[layer] = scores

                    for c, eff in scores:
                        node = f"{layer}_ch{c}"
                        circuits[cls].add_node(node, importance=eff, layer=layer)

                layers = list(cache.keys())
                #for l1, l2 in zip(layers[:-1], layers[1:]):
                #    for c1,_ in layer_topk[l1]:
                #        for c2,_ in layer_topk[l2]:
                #            n1 = f"{l1}_ch{c1}"
                #            n2 = f"{l2}_ch{c2}"

                #            base_tgt = cache[l2][:,c2].mean()
                #            patched_tgt = self.patch(x, l1, c1)[:,c2].mean()
                #            eff = (base_tgt - patched_tgt).item()

                #            if abs(eff) > 0.001:
                #                circuits[cls].add_edge(n1,n2,weight=eff)
                for l1, l2 in zip(layers[:-1], layers[1:]):
                    for c1,_ in layer_topk[l1]:
                        patched_out, patched_cache = self.patch_and_run(x, l1, c1)

                        for c2,_ in layer_topk[l2]:
                            base_tgt = cache[l2][:, c2].mean()
                            patched_tgt = patched_cache[l2][:, c2].mean()

                            eff = (base_tgt - patched_tgt).item()

                            if abs(eff) > 0.001:
                                n1 = f"{l1}_ch{c1}"
                                n2 = f"{l2}_ch{c2}"
                                circuits[cls].add_edge(n1, n2, weight=eff)
            if i >= max_batches:
                break

        return circuits

print("\n=== CIRCUIT EXTRACTION ===")
tracer_artpainting_resnet50 = AdapterCircuitTracer(model_resnet50_artpainting)
circuits_artpainting_resnet50 = tracer_artpainting_resnet50.compute_circuits(train_loader_artpainting)

# -----------------------------
# SAVE CIRCUITS
# -----------------------------
import pickle
circuit_path = os.path.join(SAVE_DIR, "circuits_resnet50_clipart.pkl")
with open(circuit_path, "wb") as f:
    pickle.dump(circuits_artpainting_resnet50, f)

print(f"✅ Circuits saved at {circuit_path}")

def visualize_layered(graph, file="circuit_layered.html"):
    net = Network(directed=True)

    layer_groups = {}
    for node, data in graph.nodes(data=True):
        layer = data.get("layer", "unknown")
        layer_groups.setdefault(layer, []).append(node)

    x_gap, y_gap = 200, 50

    for i, (layer, nodes) in enumerate(sorted(layer_groups.items())):
        for j, node in enumerate(nodes):
            net.add_node(
                node,
                x=i*x_gap,
                y=j*y_gap,
                fixed=True,
                value=abs(graph.nodes[node].get("importance", 1)),
                title=f"Layer {layer}"
            )

    for u, v, data in graph.edges(data=True):
        net.add_edge(u, v, value=abs(data.get("weight", 1)))

    net.write_html(file)   # ✅ FIX
    print(f"Saved visualization to {file}")

# visualize one class
cls = list(circuits_artpainting_resnet50.keys())[0]
print(cls)
print(circuits_artpainting_resnet50[cls])
visualize_layered(circuits_artpainting_resnet50[cls])


dataset_cartoon = datasets.ImageFolder("/home/aban/circuit_tracing/DomainNet/inforgraph", transform=transform)

generator_cartoon = torch.Generator().manual_seed(42)
train_size_cartoon = int(0.8 * len(dataset_cartoon))
test_size_cartoon = len(dataset_cartoon) - train_size_cartoon

train_dataset_cartoon, test_dataset_cartoon = random_split(dataset_cartoon, [train_size_cartoon, test_size_cartoon], generator=generator_cartoon)

train_loader_cartoon = DataLoader(train_dataset_cartoon, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
test_loader_cartoon = DataLoader(test_dataset_cartoon, batch_size=BATCH_SIZE, num_workers = 4)

model_cartoon = ResNet50_Adapter(NUM_CLASSES).to(DEVICE)

# -----------------------------
# TRAIN WITH EARLY STOPPING
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_cartoon.parameters()), lr=1e-3)

best_val = 0
patience_counter = 0


def train_epoch():
    model_cartoon.train()
    total, correct = 0,0
    for x,y in train_loader_cartoon:
        x,y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model_cartoon(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(1)
        correct += (pred==y).sum().item()
        total += y.size(0)

    return correct/total


@torch.no_grad()
def evaluate():
    model_cartoon.eval()
    total, correct = 0,0
    for x,y in test_loader_cartoon:
        x,y = x.to(DEVICE), y.to(DEVICE)
        out = model_cartoon(x)
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
        torch.save(model_cartoon.state_dict(), os.path.join(SAVE_DIR, "best_model_inforgraph_resnet50.pth"))
        print("✅ Model saved")
    else:
        break

# -----------------------------
# LOAD BEST MODEL
# -----------------------------
model_cartoon.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model_inforgraph_resnet50.pth")))
model_cartoon.eval()

print("\n=== CIRCUIT EXTRACTION ===")
tracer_cartoon_resnet50 = AdapterCircuitTracer(model_cartoon)
circuits_cartoon_resnet50 = tracer_cartoon_resnet50.compute_circuits(train_loader_cartoon)

# -----------------------------
# SAVE CIRCUITS
# -----------------------------
import pickle
circuit_path = os.path.join(SAVE_DIR, "circuits_resnet50_inforgraph.pkl")
with open(circuit_path, "wb") as f:
    pickle.dump(circuits_cartoon_resnet50, f)

print(f"✅ Circuits saved at {circuit_path}")

# visualize one class
cls = list(circuits_cartoon_resnet50.keys())[0]
print(cls)
print(circuits_cartoon_resnet50[cls])
visualize_layered(circuits_cartoon_resnet50[cls])

@torch.no_grad()
def circuit_faithfulness(model, circuits, loader, device):
    model.eval()

    results = {}

    for cls, graph in circuits.items():
        print(f"Evaluating class {cls}")

        total, correct_full, correct_pruned = 0, 0, 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # full prediction
            out_full = model(x)
            pred_full = out_full.argmax(1)

            # prune circuit neurons
            def prune_hook(layer_name, channels):
                def hook(m, i, o):
                    o = o.clone()
                    for ch in channels:
                        o[:, ch] = 0
                    return o
                return hook

            hooks = []
            layer_channels = {}

            # group nodes by layer
            for node in graph.nodes():
                layer, ch = node.split("_ch")
                layer_channels.setdefault(layer, []).append(int(ch))

            for layer, chs in layer_channels.items():
                handle = model.adapters[layer].register_forward_hook(
                    prune_hook(layer, chs)
                )
                hooks.append(handle)

            out_pruned = model(x)
            pred_pruned = out_pruned.argmax(1)

            for h in hooks:
                h.remove()

            correct_full += (pred_full == y).sum().item()
            correct_pruned += (pred_pruned == y).sum().item()
            total += y.size(0)

        acc_full = correct_full / total
        acc_pruned = correct_pruned / total

        results[cls] = {
            "full_acc": acc_full,
            "pruned_acc": acc_pruned,
            "drop": acc_full - acc_pruned
        }

        print(f"Drop: {results[cls]['drop']:.4f}")

    return results

circuit_faithfulness(model_resnet50_artpainting, circuits_artpainting_resnet50, test_loader_artpainting, "cuda")

import networkx as nx

def extract_top_paths(graph, top_k=5):
    paths = []

    # ensure DAG
    if not nx.is_directed_acyclic_graph(graph):
        graph = nx.DiGraph([(u,v,d) for u,v,d in graph.edges(data=True)])

    for source in graph.nodes():
        for target in graph.nodes():
            if source == target:
                continue

            try:
                path = nx.shortest_path(graph, source, target)
                weight = 0

                for u, v in zip(path[:-1], path[1:]):
                    weight += abs(graph[u][v].get("weight", 0))

                paths.append((path, weight))

            except:
                continue

    paths = sorted(paths, key=lambda x: x[1], reverse=True)[:top_k]
    return paths

paths = extract_top_paths(circuits_artpainting_resnet50[0], top_k=10)

for p, w in paths:
    print(" -> ".join(p), "| score:", w)

def compare_circuits(circuits_a, circuits_b):
    results = {}

    for cls in circuits_a:
        nodes_a = set(circuits_a[cls].nodes())
        nodes_b = set(circuits_b[cls].nodes())

        intersection = nodes_a & nodes_b
        union = nodes_a | nodes_b

        jaccard = len(intersection) / len(union) if len(union) > 0 else 0

        results[cls] = {
            "overlap": len(intersection),
            "jaccard": jaccard
        }

    return results

results = compare_circuits(circuits_artpainting_resnet50, circuits_cartoon_resnet50)

for cls, stats in results.items():
    print(f"Class {cls}: Jaccard {stats['jaccard']:.3f}")

dataset_photo = datasets.ImageFolder("/home/aban/circuit_tracing/DomainNet/painting", transform=transform)

generator_photo = torch.Generator().manual_seed(42)
train_size_photo = int(0.8 * len(dataset_photo))
test_size_photo = len(dataset_photo) - train_size_photo

train_dataset_photo, test_dataset_photo = random_split(dataset_photo, [train_size_photo, test_size_photo], generator=generator_photo)

train_loader_photo = DataLoader(train_dataset_photo, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
test_loader_photo = DataLoader(test_dataset_photo, batch_size=BATCH_SIZE, num_workers = 4)

model_photo = ResNet50_Adapter(NUM_CLASSES).to(DEVICE)

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
        torch.save(model_photo.state_dict(), os.path.join(SAVE_DIR, "best_model_painting_resnet50.pth"))
        print("✅ Model saved")
    else:
        break

# -----------------------------
# LOAD BEST MODEL
# -----------------------------
model_photo.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model_painting_resnet50.pth")))
model_photo.eval()

print("\n=== CIRCUIT EXTRACTION ===")
tracer_photo_resnet50 = AdapterCircuitTracer(model_photo)
circuits_photo_resnet50 = tracer_photo_resnet50.compute_circuits(train_loader_photo)

# -----------------------------
# SAVE CIRCUITS
# -----------------------------
import pickle
circuit_path = os.path.join(SAVE_DIR, "circuits_resnet50_painting.pkl")
with open(circuit_path, "wb") as f:
    pickle.dump(circuits_photo_resnet50, f)

print(f"✅ Circuits saved at {circuit_path}")



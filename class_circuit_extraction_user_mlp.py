
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from typing import Dict, Any

# -------- User's model (kept as-is) --------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3*28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        self.hidden = F.relu(self.fc1(x))  # Save hidden for tracing (post-activation)
        return self.fc2(self.hidden)

# --------- Circuit extraction utilities ----------

@torch.no_grad()
def _collect_class_stats_with_hooks(model: nn.Module,
                                    dataloader,
                                    num_classes: int,
                                    device: str = "cpu"):
    \"\"\"
    Collect class-wise E[|x|], P(a>0), E[ReLU(a)] using a forward hook on fc1 to capture pre-activations a.
    Assumes dataloader yields (x, y) with x: (N, 3, 28, 28) and y in [0..num_classes-1].
    \"\"\"
    model.eval().to(device)

    # To capture pre-activations of fc1
    preacts_bucket = []  # will store (batch_size, H)
    def hook_fc1(module, inputs, output):
        # output is 'a' = xW1 + b1 before ReLU (because hook is attached to fc1)
        preacts_bucket.append(output.detach())

    hook_handle = model.fc1.register_forward_hook(hook_fc1)

    # Prepare accumulators
    D = 3 * 28 * 28
    H = model.fc1.out_features

    class_stats = {
        c: {
            "sum_abs_x": torch.zeros(D, device="cpu"),
            "sum_active": torch.zeros(H, device="cpu"),
            "sum_relu_a": torch.zeros(H, device="cpu"),
            "count": 0
        } for c in range(num_classes)
    }

    for xb, yb in dataloader:
        xb = xb.to(device)   # (N,3,28,28)
        yb = yb.to(device)   # (N,)

        # forward pass (fills preacts_bucket with one tensor)
        _ = model(xb)
        assert len(preacts_bucket) > 0, "Forward hook did not capture pre-activations."
        a1 = preacts_bucket.pop()             # (N,H) on device
        h = F.relu(a1)                        # (N,H)

        # inputs flattened for E|x|
        xf = xb.view(xb.size(0), -1)          # (N,D)
        abs_x = xf.abs().detach().cpu()       # (N,D)
        a1_cpu = a1.detach().cpu()            # (N,H)
        h_cpu = h.detach().cpu()              # (N,H)
        active = (a1_cpu > 0).float()         # (N,H)

        for c in range(num_classes):
            mask = (yb == c).detach().cpu()
            if mask.any():
                x_c = abs_x[mask]             # (Nc, D)
                a1_c_active = active[mask]    # (Nc, H)
                h_c = h_cpu[mask]             # (Nc, H)
                Nc = x_c.shape[0]

                st = class_stats[c]
                st["sum_abs_x"] += x_c.sum(dim=0)         # (D,)
                st["sum_active"] += a1_c_active.sum(dim=0) # (H,)
                st["sum_relu_a"] += h_c.sum(dim=0)         # (H,)
                st["count"] += int(Nc)

    hook_handle.remove()

    # Turn sums into expectations / probabilities
    out = {}
    for c, st in class_stats.items():
        cnt = max(st["count"], 1)
        out[c] = {
            "E_abs_x": (st["sum_abs_x"] / cnt).numpy(),           # (D,)
            "P_active": (st["sum_active"] / cnt).numpy(),         # (H,)
            "E_relu_a": (st["sum_relu_a"] / cnt).numpy(),         # (H,)
            "count": cnt
        }
    return out

def build_class_circuits_for_user_mlp(model: nn.Module,
                                      class_stats: Dict[int, Dict[str, Any]],
                                      topk_in_per_h: int = 20,
                                      topk_h_to_out: int = 50,
                                      use_abs: bool = True):
    \"\"\"
    Builds a NetworkX DiGraph per class for the provided MLP.
    Returns: dict {class_id: nx.DiGraph}
    \"\"\"
    W1 = model.fc1.weight.detach().cpu().numpy()  # (H, D)
    b1 = model.fc1.bias.detach().cpu().numpy()    # (H,)
    W2 = model.fc2.weight.detach().cpu().numpy()  # (C, H)
    b2 = model.fc2.bias.detach().cpu().numpy()    # (C,)

    H, D = W1.shape
    C, _ = W2.shape

    circuits = {}

    for c in range(C):
        stats = class_stats[c]
        E_abs_x  = stats["E_abs_x"]    # (D,)
        P_active = stats["P_active"]   # (H,)
        E_relu_a = stats["E_relu_a"]   # (H,)

        W1_eff = np.abs(W1) if use_abs else W1
        # (D,H): multiply each column h by P_active[h] and each row i by E_abs_x[i]
        in_h_importance = (W1_eff.T * E_abs_x).T    # (H,D)
        in_h_importance = (in_h_importance.T * P_active).T  # (H,D) * (H,) -> (H,D)
        in_h_importance = in_h_importance.T        # (D,H)

        W2_eff = np.abs(W2) if use_abs else W2
        h_out_importance = W2_eff[c, :] * E_relu_a  # (H,)

        G = nx.DiGraph()
        for i in range(D): G.add_node(f"in_{i}", layer="input")
        for h in range(H): G.add_node(f"h_{h}", layer="hidden")
        G.add_node(f"out_{c}", layer="output", class_id=c)

        # Hidden -> out (top-K)
        top_h = np.argsort(-h_out_importance)[:min(topk_h_to_out, H)]
        for h in top_h:
            w = float(h_out_importance[h])
            if w > 0:
                G.add_edge(f"h_{h}", f"out_{c}", weight=w)

        # For each chosen hidden, keep its top-K parents
        for h in top_h:
            col = in_h_importance[:, h]  # (D,)
            idx = np.argsort(-col)[:min(topk_in_per_h, D)]
            for i in idx:
                w = float(col[i])
                if w > 0:
                    G.add_edge(f"in_{i}", f"h_{h}", weight=w)

        circuits[c] = G

    return circuits

def save_circuits_as_npy(circuits_dict, path="colored_mnist_class_circuits.npy"):
    serial = {}
    for c, G in circuits_dict.items():
        serial[c] = nx.readwrite.json_graph.node_link_data(G)
    np.save(path, serial, allow_pickle=True)

def load_circuits_from_npy(path="colored_mnist_class_circuits.npy"):
    serial = np.load(path, allow_pickle=True).item()
    circuits = {}
    for c, data in serial.items():
        circuits[c] = nx.readwrite.json_graph.node_link_graph(data, directed=True)
    return circuits

# -------- Convenience "runner" (optional) --------
def extract_and_save_circuits(model: nn.Module,
                              dataloader,
                              num_classes: int = 10,
                              device: str = "cpu",
                              topk_in_per_h: int = 20,
                              topk_h_to_out: int = 50,
                              save_path: str = "colored_mnist_class_circuits.npy"):
    \"\"\"
    One-call pipeline: collect stats -> build graphs -> save .npy
    Returns (save_path, stats_dict)
    \"\"\"
    stats = _collect_class_stats_with_hooks(model, dataloader, num_classes, device=device)
    circuits = build_class_circuits_for_user_mlp(
        model, stats,
        topk_in_per_h=topk_in_per_h,
        topk_h_to_out=topk_h_to_out,
        use_abs=True
    )
    save_circuits_as_npy(circuits, save_path)
    return save_path, stats

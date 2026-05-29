
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
import os

# Constants
BATCH_SIZE = 64
SEED = 42
NUM_CLASSES = 10
TOP_K = 10
SAVE_DIR = "./graphs"

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(SAVE_DIR, exist_ok=True)

# Colored MNIST by tinting grayscale digits
class ColoredMNIST(datasets.MNIST):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        img = img.repeat(3, 1, 1)  # Convert to 3 channels
        color = torch.zeros_like(img)
        color[target % 3] = img[target % 3]
        return color, target

# Load Colored MNIST
transform = transforms.Compose([transforms.ToTensor()])
dataset = ColoredMNIST(root='./data', train=True, download=True, transform=transform)
loaders = {}
for c in range(NUM_CLASSES):
    indices = [i for i, (_, label) in enumerate(dataset) if label == c][:TOP_K * 2]
    loaders[c] = DataLoader(Subset(dataset, indices), batch_size=len(indices), shuffle=False)

# MLP for 3x28x28 input
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(3 * 28 * 28, 128)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        a1 = self.relu(self.l1(x))
        return self.l2(a1)

# Attribution
def compute_attribution_matrix(model, layer1, layer2, x):
    model.eval()
    attr = torch.zeros(layer1.out_features, layer2.out_features)
    x = x.view(x.size(0), -1)
    x.requires_grad = True
    a1 = model.relu(layer1(x))
    for i in range(layer1.out_features):
        grad_outputs = torch.zeros_like(a1)
        grad_outputs[:, i] = 1
        a1.backward(grad_outputs, retain_graph=True)
        grad = model.l2.weight.grad[:, i]
        attr[i] += grad.abs()
        model.zero_grad()
    return attr / x.size(0)

def build_circuit(attr_matrix, k=TOP_K):
    return {
        "l1": torch.topk(attr_matrix.sum(dim=1), k).indices.tolist(),
        "l2": torch.topk(attr_matrix.sum(dim=0), k).indices.tolist()
    }

def circuit_to_graph(circuit, attr_matrix):
    G = nx.DiGraph()
    for i in circuit["l1"]:
        for j in circuit["l2"]:
            G.add_edge(f"l1_{i}", f"l2_{j}", weight=attr_matrix[i, j].item())
    return G

def save_graph(G, class_id):
    pos = nx.spring_layout(G)
    centrality = nx.betweenness_centrality(G)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_color=list(centrality.values()),
            cmap=plt.cm.plasma, edge_color=weights, width=2.0, edge_cmap=plt.cm.Blues)
    plt.title(f"Class {class_id} Circuit")
    plt.savefig(f"{SAVE_DIR}/class_{class_id}_graph.png")
    plt.clf()

def main():
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Light training
    for x, y in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        break

    graphs = {}
    for c in range(NUM_CLASSES):
        x, _ = next(iter(loaders[c]))
        attr_matrix = compute_attribution_matrix(model, model.l1, model.l2, x)
        circuit = build_circuit(attr_matrix)
        G = circuit_to_graph(circuit, attr_matrix)
        graphs[c] = G
        save_graph(G, c)

    # Compare classwise circuits using graph isomorphism & betweenness
    for i in range(NUM_CLASSES):
        for j in range(i + 1, NUM_CLASSES):
            iso = nx.is_isomorphic(graphs[i], graphs[j])
            c1 = nx.betweenness_centrality(graphs[i])
            c2 = nx.betweenness_centrality(graphs[j])
            diff = sum(abs(c1.get(n, 0) - c2.get(n, 0)) for n in set(c1) | set(c2))
            print(f"Class {i} vs {j}: Isomorphic={iso}, Betweenness Diff={diff:.4f}")

if __name__ == "__main__":
    main()

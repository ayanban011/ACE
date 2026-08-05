# PACS DG: Sequential Methods + Explicit Train/Test Domains + Per-Domain Accuracy

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -------- CONFIG --------
DATA_ROOT = "./OfficeHomeDataset_10072016"
METHODS = ["ERM", "IRM", "CORAL", "MIXUP", "DANN"]
DOMAINS = ["Art", "Clipart", "Product", "Real_World"]
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- DATA --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def get_domain_loaders(test_domain):
    train_domains = [d for d in DOMAINS if d != test_domain]

    train_loaders = {
        d: DataLoader(
            datasets.ImageFolder(os.path.join(DATA_ROOT, d), transform),
            batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=16
        ) for d in train_domains
    }

    test_loader = DataLoader(
        datasets.ImageFolder(os.path.join(DATA_ROOT, test_domain), transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=16
    )

    return train_loaders, test_loader

# -------- GRL --------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

# -------- MODEL --------
class Model(nn.Module):
    def __init__(self, num_classes, num_domains):
        super().__init__()
        base = models.vgg19(pretrained=True)
        self.features = base.features
        self.avgpool = base.avgpool
        self.feat_dim = 512*7*7

        self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.domain_clf = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_domains)
        )

    def forward(self, x, grl_lambda=0.0):
        x = self.features(x)
        x = self.avgpool(x)
        feat = torch.flatten(x,1)
        logits = self.classifier(feat)
        dom_logits = self.domain_clf(GradReverse.apply(feat, grl_lambda))
        return logits, feat, dom_logits

# class Model(nn.Module):
#     def __init__(self, num_classes, num_domains):
#         super().__init__()

#         base = models.resnet50(pretrained=True)

#         # Extract all layers except final FC
#         self.features = nn.Sequential(*list(base.children())[:-1])

#         self.feat_dim = 2048

#         self.classifier = nn.Linear(self.feat_dim, num_classes)

#         self.domain_clf = nn.Sequential(
#             nn.Linear(self.feat_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, num_domains)
#         )

#     def forward(self, x, grl_lambda=0.0):
#         x = self.features(x)        # [B, 2048, 1, 1]
#         feat = torch.flatten(x, 1)  # [B, 2048]

#         logits = self.classifier(feat)

#         dom_logits = self.domain_clf(
#             GradReverse.apply(feat, grl_lambda)
#         )

#         return logits, feat, dom_logits

# class Model(nn.Module):
#     def __init__(self, num_classes, num_domains):
#         super().__init__()

#         base = models.mobilenet_v2(pretrained=True)

#         # Backbone
#         self.features = base.features

#         # MobileNetV2 uses global avg pooling implicitly
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))

#         self.feat_dim = 1280  # IMPORTANT

#         # Replace classifier
#         self.classifier = nn.Linear(self.feat_dim, num_classes)

#         self.domain_clf = nn.Sequential(
#             nn.Linear(self.feat_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, num_domains)
#         )

#     def forward(self, x, grl_lambda=0.0):
#         x = self.features(x)          # [B, 1280, H, W]
#         x = self.pool(x)              # [B, 1280, 1, 1]
#         feat = torch.flatten(x, 1)    # [B, 1280]

#         logits = self.classifier(feat)

#         dom_logits = self.domain_clf(
#             GradReverse.apply(feat, grl_lambda)
#         )

#         return logits, feat, dom_logits

# class Model(nn.Module):
#     def __init__(self, num_classes, num_domains):
#         super().__init__()

#         base = models.vit_b_16(pretrained=True)

#         # Remove classification head
#         base.heads = nn.Identity()

#         self.backbone = base
#         self.feat_dim = 768  # IMPORTANT

#         self.classifier = nn.Linear(self.feat_dim, num_classes)

#         self.domain_clf = nn.Sequential(
#             nn.Linear(self.feat_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, num_domains)
#         )

#     def forward(self, x, grl_lambda=0.0):
#         feat = self.backbone(x)   # [B, 768]

#         logits = self.classifier(feat)

#         dom_logits = self.domain_clf(
#             GradReverse.apply(feat, grl_lambda)
#         )

#         return logits, feat, dom_logits

# -------- LOSSES --------
def irm_penalty(logits, y):
    scale = torch.tensor(1.).to(DEVICE).requires_grad_()
    loss = F.cross_entropy(logits * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)


def coral(x, y):
    xm = x - x.mean(0, keepdim=True)
    ym = y - y.mean(0, keepdim=True)
    return torch.mean((xm.t()@xm - ym.t()@ym)**2)


def mixup(x, y, alpha=0.2):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(x.size(0)).to(DEVICE)
    x_mix = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return x_mix, y_a, y_b, lam

# -------- TRAIN --------
def train(model, loaders, optimizer, method):
    model.train()
    domain_names = list(loaders.keys())
    iters = min(len(loaders[d]) for d in domain_names)
    iters_dict = {d: iter(loaders[d]) for d in domain_names}

    for _ in range(iters):
        optimizer.zero_grad()

        cls_losses, irm_losses, dom_losses = [], [], []
        feats = []

        for d_idx, d in enumerate(domain_names):
            x, y = next(iters_dict[d])
            x, y = x.to(DEVICE), y.to(DEVICE)

            if method == "MIXUP":
                x, y_a, y_b, lam = mixup(x, y)
                logits, feat, dom_logits = model(x)
                cls_loss = lam * F.cross_entropy(logits, y_a) + (1-lam)*F.cross_entropy(logits, y_b)
            else:
                logits, feat, dom_logits = model(x, grl_lambda=1.0 if method=="DANN" else 0.0)
                cls_loss = F.cross_entropy(logits, y)

            cls_losses.append(cls_loss)
            feats.append(feat)

            if method == "IRM":
                irm_losses.append(irm_penalty(logits, y))

            if method == "DANN":
                dom_labels = torch.full((len(x),), d_idx, dtype=torch.long).to(DEVICE)
                dom_losses.append(F.cross_entropy(dom_logits, dom_labels))

        loss = torch.stack(cls_losses).mean()

        if method == "IRM":
            loss += torch.stack(irm_losses).mean()

        if method == "CORAL":
            coral_loss = 0
            for i in range(len(feats)-1):
                coral_loss += coral(feats[i], feats[i+1])
            loss += coral_loss

        if method == "DANN":
            loss += torch.stack(dom_losses).mean()

        loss.backward()
        optimizer.step()

# -------- EVAL --------
def evaluate(model, loader):
    model.eval()
    correct,total = 0,0

    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            logits,_,_ = model(x)
            preds = logits.argmax(1)
            correct += (preds==y).sum().item()
            total += y.size(0)

    return correct/total

# -------- MAIN --------
def run():
    results = {}

    for method in METHODS:
        print(f"\n===== METHOD: {method} =====")
        results[method] = {}

        for test_domain in DOMAINS:
            print(f"\n--- Test Domain: {test_domain} ---")

            train_loaders, test_loader = get_domain_loaders(test_domain)
            num_classes = len(next(iter(train_loaders.values())).dataset.classes)

            model = Model(num_classes, num_domains=len(train_loaders)).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=LR)

            for epoch in range(EPOCHS):
                train(model, train_loaders, optimizer, method)

            acc = evaluate(model, test_loader)
            results[method][test_domain] = acc
            print(f"Final Acc ({test_domain}): {acc:.4f}")

    print("\n===== FINAL RESULTS =====")
    for method in results:
        print(f"\n{method}:")
        for d in results[method]:
            print(f"  {d}: {results[method][d]:.4f}")

if __name__ == "__main__":
    run()

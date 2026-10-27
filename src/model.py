"""1D ResNet for 12-lead ECG + Grad-CAM."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 7, stride, 3, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 7, 1, 3, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.short = (nn.Identity() if (cin == cout and stride == 1)
                      else nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)))

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return F.relu(o + self.short(x))


class ECGResNet(nn.Module):
    def __init__(self, n_leads=12, n_classes=5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(n_leads, 32, 15, 1, 7, bias=False), nn.BatchNorm1d(32), nn.ReLU())
        self.layers = nn.Sequential(Block(32, 32, 1), Block(32, 64, 2), Block(64, 128, 2), Block(128, 256, 2))
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(256, n_classes)

    def features(self, x):
        return self.layers(self.stem(x))

    def forward(self, x):
        return self.head(self.drop(self.features(x).mean(-1)))


def normalize(X, mean, std):
    """X: (N, T, 12) raw -> (N, 12, T) standardised float32."""
    return ((X - mean) / std).transpose(0, 2, 1).astype(np.float32)


@torch.no_grad()
def predict(model, Xn, device="cpu", bs=256):
    model.eval()
    out = []
    for i in range(0, len(Xn), bs):
        xb = torch.from_numpy(Xn[i:i + bs]).to(device)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


def gradcam_1d(model, x, cls, device="cpu"):
    """x: (12, T) normalised numpy array. Returns (probability, cam of length T scaled to 0-1)."""
    model.eval()
    xt = torch.from_numpy(x[None]).float().to(device)
    feats = model.features(xt)
    feats.retain_grad()
    logits = model.head(feats.mean(-1))
    model.zero_grad()
    logits[0, cls].backward()
    w = feats.grad.mean(-1, keepdim=True)
    cam = F.relu((w * feats).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-1], mode="linear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    cam = cam / (cam.max() + 1e-8)
    return float(torch.sigmoid(logits.detach())[0, cls]), cam

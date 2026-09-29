from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


STATE_DIM = 9
ACTION_DIM = 2
EP_LEN = 28


class HistoryWorldModel(nn.Module):
    """Predict next robot/object state from short motion/action history."""

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        in_dim = STATE_DIM * 3 + ACTION_DIM * 3
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.ee_head = nn.Linear(hidden, 2)
        self.object_head = nn.Linear(hidden, 2)
        self.joint_head = nn.Linear(hidden, 5)
        self.contact_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        contact_logit = self.contact_head(h).squeeze(-1)
        contact = torch.sigmoid(contact_logit).unsqueeze(-1)
        delta = torch.cat(
            [
                self.ee_head(h),
                contact * self.object_head(h),
                self.joint_head(h),
            ],
            dim=-1,
        )
        return delta, contact_logit


def history_feature(s, s1, s2, a1, a2, a):
    return np.concatenate(
        [s, s-s1, s1-s2, a1, a2, a], axis=-1
    ).astype(np.float32)


def make_training(s, a, ns):
    features, targets, gates = [], [], []
    object_motion = np.linalg.norm(ns[:, 2:4] - s[:, 2:4], axis=1)
    threshold = max(0.0007, float(np.percentile(object_motion, 55)))

    for t in range(2, len(s)):
        if t % EP_LEN < 2:
            continue
        features.append(
            history_feature(s[t], s[t-1], s[t-2], a[t-1], a[t-2], a[t])
        )
        targets.append(ns[t] - s[t])
        gates.append(float(object_motion[t] > threshold))

    return (
        np.stack(features).astype(np.float32),
        np.stack(targets).astype(np.float32),
        np.asarray(gates, np.float32),
        threshold,
    )


def train_world_model(s, a, ns, device, epochs: int = 35):
    x, y, g, threshold = make_training(s, a, ns)
    xm = x.mean(0).astype(np.float32)
    xs = (x.std(0) + 1e-5).astype(np.float32)
    ys = (y.std(0) + 1e-5).astype(np.float32)

    xt = torch.tensor((x-xm)/xs, device=device)
    yt = torch.tensor(y/ys, device=device)
    gt = torch.tensor(g, device=device)

    model = HistoryWorldModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.35e-3, weight_decay=1e-5)

    pos = max(float(g.sum()), 1.0)
    neg = max(float(len(g)-g.sum()), 1.0)
    pos_weight = torch.tensor(min(neg/pos, 8.0), device=device)

    n = len(xt)
    ids_all = torch.arange(n, device=device)
    batch = 512

    print(
        f"history windows={n} contact_threshold={threshold*1000:.2f}mm "
        f"positives={g.mean():.1%}"
    )

    for epoch in range(epochs):
        order = ids_all[torch.randperm(n, device=device)]
        total = 0.0
        for start in range(0, n, batch):
            ids = order[start:start+batch]
            pred, contact_logit = model(xt[ids])
            target = yt[ids]
            gate = gt[ids]

            ee_loss = F.mse_loss(pred[:, :2], target[:, :2])
            object_err = ((pred[:, 2:4]-target[:, 2:4])**2).mean(1)
            object_loss = ((1.0 + 8.0*gate) * object_err).mean()
            joint_loss = F.mse_loss(pred[:, 4:], target[:, 4:])
            contact_loss = F.binary_cross_entropy_with_logits(
                contact_logit, gate, pos_weight=pos_weight
            )
            loss = (
                0.30*ee_loss
                + 2.8*object_loss
                + 0.22*joint_loss
                + 0.75*contact_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * len(ids)

        if epoch in {0, 4, 9, 19, 29, epochs-1}:
            print(f"epoch {epoch+1:02d}/{epochs} loss={total/n:.5f}")

    stats = {
        "xm": torch.tensor(xm, device=device),
        "xs": torch.tensor(xs, device=device),
        "ys": torch.tensor(ys, device=device),
    }
    return model.eval(), stats


def torch_feature(s, s1, s2, a1, a2, a):
    return torch.cat([s, s-s1, s1-s2, a1, a2, a], dim=-1)


@torch.inference_mode()
def predict(model, stats, s, s1, s2, a1, a2, a):
    x = torch_feature(s, s1, s2, a1, a2, a)
    xn = (x-stats["xm"]) / stats["xs"]
    dn, contact_logit = model(xn)
    ns = s + dn * stats["ys"]
    return ns, torch.sigmoid(contact_logit)

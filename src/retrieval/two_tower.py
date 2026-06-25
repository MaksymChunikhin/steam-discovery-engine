import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ItemTower(nn.Module):
    # Two branches: dense review embedding and structured metadata, fused into one vector
    # Две ветки: плотный эмбеддинг отзывов и структурные метаданные, сливаются в один вектор
    def __init__(self, emb_dim, struct_dim, out_dim=64, hidden_emb=128,
                 hidden_struct=32, dropout=0.2):
        super().__init__()
        self.emb_branch = nn.Sequential(nn.Linear(emb_dim, hidden_emb), nn.ReLU(), nn.Dropout(dropout))
        self.struct_branch = nn.Sequential(nn.Linear(struct_dim, hidden_struct), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden_emb + hidden_struct, out_dim)

    def forward(self, emb, struct):
        x = torch.cat([self.emb_branch(emb), self.struct_branch(struct)], dim=1)
        return F.normalize(self.head(x), dim=1)


class UserTower(nn.Module):
    # Two branches: genre/tag history profile and numeric stats
    # Две ветки: профиль истории по жанрам/тегам и числовые статистики
    def __init__(self, hist_dim, num_dim, out_dim=64, hidden_hist=32,
                 hidden_num=16, dropout=0.2):
        super().__init__()
        self.hist_branch = nn.Sequential(nn.Linear(hist_dim, hidden_hist), nn.ReLU(), nn.Dropout(dropout))
        self.num_branch = nn.Sequential(nn.Linear(num_dim, hidden_num), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden_hist + hidden_num, out_dim)

    def forward(self, hist, num):
        x = torch.cat([self.hist_branch(hist), self.num_branch(num)], dim=1)
        return F.normalize(self.head(x), dim=1)


class TwoTower(nn.Module):
    def __init__(self, emb_dim, struct_dim, hist_dim, num_dim, out_dim=64,
                 temperature=0.05, dropout=0.2):
        super().__init__()
        self.item_tower = ItemTower(emb_dim, struct_dim, out_dim, dropout=dropout)
        self.user_tower = UserTower(hist_dim, num_dim, out_dim, dropout=dropout)
        self.logit_scale = 1.0 / temperature


def _eval_recall(model, item_emb, item_struct, user_hist, user_num,
                 eval_rows, eval_truth, eval_seen, k=10):
    # Recall@k for monitoring; eval_seen items are masked out (already-seen games)
    # Recall@k для мониторинга; виденные игры (eval_seen) маскируются
    model.eval()
    with torch.no_grad():
        I = model.item_tower(item_emb, item_struct)
        U = model.user_tower(user_hist[eval_rows], user_num[eval_rows])
        scores = (U @ I.T).cpu().numpy()
    hits = 0
    for r in range(len(eval_rows)):
        s = scores[r]
        seen = eval_seen[r]
        if len(seen):
            s[seen] = -1e9
        top = np.argpartition(-s, k)[:k]
        if eval_truth[r] in top:
            hits += 1
    return hits / len(eval_rows)


def train_two_tower(data, epochs=10, batch_size=1024, lr=1e-3, dropout=0.2,
                    temperature=0.05, out_dim=64, device="cuda",
                    eval_data=None, eval_every=2, k=10, seed=42, log=print):
    # Train the Two-Tower model with in-batch negatives (softmax cross-entropy).
    # User features come from training history only; eval_data (if given) is used
    # purely to monitor Recall@10 over epochs, never for training or selection.
    # Обучаем Two-Tower с in-batch negatives (softmax cross-entropy).
    # Признаки юзера — только из train-истории; eval_data (если задан) служит лишь
    # для мониторинга Recall@10 по эпохам, не для обучения и не для отбора.
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    item_emb = torch.tensor(data["item_emb"], device=device)
    item_struct = torch.tensor(data["item_struct"], device=device)
    user_hist = torch.tensor(data["user_hist"], device=device)
    user_num = torch.tensor(data["user_num"], device=device)
    pairs = data["pairs"]

    model = TwoTower(item_emb.shape[1], item_struct.shape[1], user_hist.shape[1],
                     user_num.shape[1], out_dim=out_dim, temperature=temperature,
                     dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = len(pairs)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n)
        total, nb = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            bu = torch.tensor(pairs[idx, 0], device=device)
            bi = torch.tensor(pairs[idx, 1], device=device)
            u = model.user_tower(user_hist[bu], user_num[bu])
            it = model.item_tower(item_emb[bi], item_struct[bi])
            logits = (u @ it.T) * model.logit_scale
            labels = torch.arange(len(idx), device=device)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        avg = total / nb
        rec = None
        if eval_data is not None and (epoch % eval_every == 0 or epoch == epochs):
            rec = _eval_recall(model, item_emb, item_struct, user_hist, user_num,
                               *eval_data, k=k)
        history.append({"epoch": epoch, "loss": avg, "recall@10": rec})
        log(f"epoch {epoch:2d}  loss {avg:.4f}" + (f"  test recall@10 {rec:.4f}" if rec is not None else ""))

    return model, history

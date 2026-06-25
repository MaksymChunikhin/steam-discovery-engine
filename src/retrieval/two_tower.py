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


def _val_recall(model, item_emb, item_struct, user_hist, user_num,
                val_users, val_truth, seen_train, k=10):
    # Recall@10 on a small validation set carved from train (monitoring only)
    # Recall@10 на маленьком val, выделенном из train (только для мониторинга)
    model.eval()
    with torch.no_grad():
        I = model.item_tower(item_emb, item_struct)
        U = model.user_tower(user_hist[val_users], user_num[val_users])
        scores = (U @ I.T).cpu().numpy()
    hits = 0
    for r, u in enumerate(val_users):
        s = scores[r]
        seen = seen_train.get(int(u))
        if seen:
            s[list(seen)] = -1e9
        top = np.argpartition(-s, k)[:k]
        if val_truth[int(u)] in top:
            hits += 1
    return hits / len(val_users)


def train_two_tower(data, epochs=10, batch_size=1024, lr=1e-3, dropout=0.2,
                    temperature=0.05, out_dim=64, device="cuda",
                    val_users=5000, eval_every=2, k=10, seed=42, log=print):
    # Train the Two-Tower model with in-batch negatives (softmax cross-entropy).
    # User features come from training history only (no test leakage at evaluation).
    # Обучаем Two-Tower с in-batch negatives (softmax cross-entropy).
    # Признаки юзера — только из train-истории (на оценке утечки нет).
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    item_emb = torch.tensor(data["item_emb"], device=device)
    item_struct = torch.tensor(data["item_struct"], device=device)
    user_hist = torch.tensor(data["user_hist"], device=device)
    user_num = torch.tensor(data["user_num"], device=device)
    pairs = data["pairs"]

    # Carve a small validation set: hold out one interaction per sampled user
    # Выделяем маленький val: по одному взаимодействию у выбранных юзеров
    items_by_user = {}
    for ui, it in pairs:
        items_by_user.setdefault(int(ui), []).append(int(it))
    eligible = [u for u, its in items_by_user.items() if len(its) >= 2]
    val_u = rng.choice(eligible, size=min(val_users, len(eligible)), replace=False)
    val_truth, val_set = {}, set()
    for u in val_u:
        it = int(rng.choice(items_by_user[u]))
        val_truth[int(u)] = it
        val_set.add((int(u), it))
    seen_train = {u: (set(items_by_user[u]) - {val_truth[u]}) for u in val_truth}
    val_u = np.array([int(u) for u in val_u])

    keep = np.array([(int(ui), int(it)) not in val_set for ui, it in pairs])
    train_pairs = pairs[keep]

    model = TwoTower(item_emb.shape[1], item_struct.shape[1], user_hist.shape[1],
                     user_num.shape[1], out_dim=out_dim, temperature=temperature,
                     dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = len(train_pairs)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n)
        total, nb = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            bu = torch.tensor(train_pairs[idx, 0], device=device)
            bi = torch.tensor(train_pairs[idx, 1], device=device)
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
        if epoch % eval_every == 0 or epoch == epochs:
            rec = _val_recall(model, item_emb, item_struct, user_hist, user_num,
                              val_u, val_truth, seen_train, k=k)
        history.append({"epoch": epoch, "loss": avg, "recall@10": rec})
        log(f"epoch {epoch:2d}  loss {avg:.4f}" + (f"  val recall@10 {rec:.4f}" if rec is not None else ""))

    return model, history

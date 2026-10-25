"""Risk models on a ``Cohort``. Every model has ``fit(cohort, idx)`` and ``risk(cohort, idx)`` (higher = worse)."""
import numpy as np
import torch
from sklearn.decomposition import PCA
from sksurv.linear_model import CoxPHSurvivalAnalysis


def _standardise(train: np.ndarray, other: np.ndarray | None = None):
    mean, std = train.mean(0), train.std(0)
    std = np.where(std < 1e-8, 1.0, std)
    return ((train - mean) / std,) if other is None else ((train - mean) / std, (other - mean) / std)


class ClinicalCox:
    """Age and sex only: the floor the image models have to beat."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha

    def _x(self, c, idx):
        return np.column_stack([c.age[idx], c.male[idx]])

    def fit(self, c, idx):
        self.mean, self.std = self._x(c, idx).mean(0), self._x(c, idx).std(0)
        self.model = CoxPHSurvivalAnalysis(alpha=self.alpha).fit((self._x(c, idx) - self.mean) / self.std, c.y[idx])
        return self

    def risk(self, c, idx):
        return self.model.predict((self._x(c, idx) - self.mean) / self.std)


class PooledCox:
    """Mean node embedding per patient -> PCA -> ridge Cox, optionally with the hand-made node numbers."""

    def __init__(self, n_components: int = 8, alpha: float = 10.0, hand: bool = False) -> None:
        self.n_components, self.alpha, self.hand = n_components, alpha, hand

    def _pooled(self, c, idx):
        return np.stack([c.bags[i].mean(0) for i in idx])

    def _x(self, c, idx, fit: bool):
        pooled = self._pooled(c, idx)
        if fit:
            self.emb_mean, self.emb_std = pooled.mean(0), np.where(pooled.std(0) < 1e-8, 1.0, pooled.std(0))
            self.pca = PCA(self.n_components, random_state=0).fit((pooled - self.emb_mean) / self.emb_std)
        z = self.pca.transform((pooled - self.emb_mean) / self.emb_std)
        cols = [z, c.age[idx, None], c.male[idx, None]] + ([c.hand_features()[idx]] if self.hand else [])
        return np.column_stack(cols)

    def fit(self, c, idx):
        x = self._x(c, idx, fit=True)
        self.mean, self.std = x.mean(0), np.where(x.std(0) < 1e-8, 1.0, x.std(0))
        self.model = CoxPHSurvivalAnalysis(alpha=self.alpha).fit((x - self.mean) / self.std, c.y[idx])
        return self

    def risk(self, c, idx):
        return self.model.predict((self._x(c, idx, fit=False) - self.mean) / self.std)


def cox_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood (Breslow ties), full batch."""
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]
    log_risk_set = torch.logcumsumexp(risk, 0)
    return -((risk - log_risk_set) * event).sum() / event.sum().clamp(min=1)


class _Net(torch.nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_attn: int, n_extra: int, dropout: float) -> None:
        super().__init__()
        self.drop = torch.nn.Dropout(dropout)
        self.embed = torch.nn.Sequential(torch.nn.Linear(d_in, d_hidden), torch.nn.GELU(), torch.nn.Dropout(dropout))
        self.attn_v = torch.nn.Linear(d_hidden, d_attn); self.attn_u = torch.nn.Linear(d_hidden, d_attn)
        self.attn_w = torch.nn.Linear(d_attn, 1)
        self.head = torch.nn.Linear(d_hidden + n_extra, 1)

    def forward(self, x, bag, pos, n_bags, n_max, extra):
        """``x`` holds the real nodes of all bags back to back; bag / pos give each node's bag and its place in it."""
        h_flat = self.embed(self.drop(x))                                  # (M, H), only real nodes
        h = h_flat.new_zeros(n_bags, n_max, h_flat.shape[1]); h[bag, pos] = h_flat
        mask = h.new_zeros(n_bags, n_max, dtype=torch.bool); mask[bag, pos] = True
        a = self.attn_w(torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h))).squeeze(-1)
        a = torch.softmax(a.masked_fill(~mask, -1e9), dim=1)
        pooled = (a.unsqueeze(-1) * h).sum(1)
        return self.head(torch.cat([pooled, extra], 1)).squeeze(-1), a


class AttentionMIL:
    """Gated-attention multiple-instance model on the node embeddings with a Cox loss.

    Full-batch training, fixed number of steps (no tuning on the folds), average of ``n_models`` seeds.
    Age and sex enter next to the pooled image representation.
    """

    def __init__(self, hidden: int = 32, attn: int = 16, dropout: float = 0.5, weight_decay: float = 1e-2,
                 lr: float = 1e-3, steps: int = 200, n_models: int = 5, use_clinical: bool = True,
                 device: str | None = None) -> None:
        self.hidden, self.attn, self.dropout, self.weight_decay = hidden, attn, dropout, weight_decay
        self.lr, self.steps, self.n_models, self.use_clinical = lr, steps, n_models, use_clinical
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _tensors(self, c, idx):
        sizes = np.array([len(c.bags[i]) for i in idx])
        x = (np.concatenate([c.bags[i] for i in idx]) - self.mean) / self.std
        bag = np.repeat(np.arange(len(idx)), sizes); pos = np.concatenate([np.arange(s) for s in sizes])
        extra = (np.column_stack([c.age[idx], c.male[idx]]) - self.extra_mean) / self.extra_std
        if not self.use_clinical:
            extra = extra[:, :0]
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=self.device)
        i64 = lambda a: torch.as_tensor(a, dtype=torch.long, device=self.device)
        return t(x), i64(bag), i64(pos), len(idx), int(sizes.max()), t(extra)

    def fit(self, c, idx):
        allnodes = np.concatenate([c.bags[i] for i in idx])
        self.mean = allnodes.mean(0); self.std = np.where(allnodes.std(0) < 1e-8, 1.0, allnodes.std(0))
        clin = np.column_stack([c.age[idx], c.male[idx]])
        self.extra_mean, self.extra_std = clin.mean(0), np.where(clin.std(0) < 1e-8, 1.0, clin.std(0))
        x, bag, pos, n_bags, n_max, extra = self._tensors(c, idx)
        time = torch.as_tensor(c.y["time"][idx], dtype=torch.float32, device=self.device)
        event = torch.as_tensor(c.y["event"][idx], dtype=torch.float32, device=self.device)
        self.nets = []
        for seed in range(self.n_models):
            torch.manual_seed(seed)
            net = _Net(x.shape[1], self.hidden, self.attn, extra.shape[1], self.dropout).to(self.device)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            for _ in range(self.steps):
                net.train(); opt.zero_grad()
                loss = cox_loss(net(x, bag, pos, n_bags, n_max, extra)[0], time, event); loss.backward(); opt.step()
            self.nets.append(net.eval())
        return self

    @torch.no_grad()
    def risk(self, c, idx):
        args = self._tensors(c, idx)
        r = torch.stack([net(*args)[0] for net in self.nets]).mean(0)
        return r.cpu().numpy()

    @torch.no_grad()
    def attention(self, c, idx):
        """Attention weight of every node (mean over the models), one array per patient."""
        args = self._tensors(c, idx)
        a = torch.stack([net(*args)[1] for net in self.nets]).mean(0).cpu().numpy()
        return [a[k, : len(c.bags[i])] for k, i in enumerate(idx)]

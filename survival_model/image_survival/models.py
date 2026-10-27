"""Risk models on a ``Cohort``. Every model has ``fit(cohort, idx)`` and ``risk(cohort, idx)`` (higher = worse)."""
from typing import ClassVar

import numpy as np
import torch
from sklearn.decomposition import PCA
from sksurv.linear_model import CoxPHSurvivalAnalysis

from survival_model.features import TabularEncoder

# Reference levels of configs/features/default.yaml
REFERENCE = {"sex": "M", "location": "left", "morphology_group": "adenocarcinoma", "grade_group": "low", "pt_group": "T0-2",
             "pn_group": "N0", "m_group": "M0", "neoadjuvant": "no", "mmr": "pMMR", "kras": "wt", "nras": "wt", "braf": "wt"}
TNM = {"numeric": ["age"], "categorical": ["sex", "pt_group", "pn_group", "m_group"]}
FULL = {"numeric": ["age", "nodes_examined"], "categorical": ["sex", "location", "morphology_group", "grade_group", "pt_group",
            "pn_group", "m_group", "neoadjuvant", "mmr", "kras", "nras", "braf"]}


def _standardise(train: np.ndarray, other: np.ndarray | None = None):
    mean, std = train.mean(0), train.std(0)
    std = np.where(std < 1e-8, 1.0, std)
    return ((train - mean) / std,) if other is None else ((train - mean) / std, (other - mean) / std)


class ExcelCox:
    """Reference from the Excel sheet: the encoder and ridge Cox of the first survival model, on the same patients and folds."""

    def __init__(self, spec: dict, alpha: float = 0.1) -> None:
        self.spec, self.alpha = spec, alpha

    def fit(self, c, idx):
        self.enc = TabularEncoder(**self.spec, reference=REFERENCE, min_category_count=10, scale_numeric=True).fit(c.features.iloc[idx])
        self.model = CoxPHSurvivalAnalysis(alpha=self.alpha).fit(self.enc.transform(c.features.iloc[idx]).to_numpy(), c.y[idx])
        return self

    def risk(self, c, idx):
        return self.model.predict(self.enc.transform(c.features.iloc[idx]).to_numpy())


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

    def __init__(self, n_components: int = 8, alpha: float = 10.0, hand: bool = False, tnm: bool = False) -> None:
        self.n_components, self.alpha, self.hand, self.tnm = n_components, alpha, hand, tnm

    def _pooled(self, c, idx):
        return np.stack([c.bags[i].mean(0) for i in idx])

    def _x(self, c, idx, fit: bool):
        pooled = self._pooled(c, idx)
        if self.tnm:  # pT, pN and M next to the image features: do the images add to the staging?
            if fit:
                self.enc = TabularEncoder(numeric=[], categorical=TNM["categorical"][1:], reference=REFERENCE, min_category_count=10).fit(c.features.iloc[idx])
            tnm = self.enc.transform(c.features.iloc[idx]).to_numpy()
        if fit:
            self.emb_mean, self.emb_std = pooled.mean(0), np.where(pooled.std(0) < 1e-8, 1.0, pooled.std(0))
            self.pca = PCA(self.n_components, random_state=0).fit((pooled - self.emb_mean) / self.emb_std)
        z = self.pca.transform((pooled - self.emb_mean) / self.emb_std)
        cols = [z, c.age[idx, None], c.male[idx, None]] + ([c.hand_features()[idx]] if self.hand else []) + ([tnm] if self.tnm else [])
        return np.column_stack(cols)

    def fit(self, c, idx):
        x = self._x(c, idx, fit=True)
        self.mean, self.std = x.mean(0), np.where(x.std(0) < 1e-8, 1.0, x.std(0))
        self.model = CoxPHSurvivalAnalysis(alpha=self.alpha).fit((x - self.mean) / self.std, c.y[idx])
        return self

    def risk(self, c, idx):
        return self.model.predict((self._x(c, idx, fit=False) - self.mean) / self.std)


class NodeRatioCox:
    """Lymph-node ratio from the images.

    A node detector scores every node (hazard h_i, P(node is tumour) = 1 - exp(-h_i)); a patient is positive when any node
    is (P = 1 - exp(-sum h_i)), so it can be trained with the patient-level "has a positive node" label only. The patient
    features are the expected number of positive nodes, their share of all nodes (the ratio), the highest node score and the
    node count, which go into a ridge Cox model with age and sex. Scores of the training patients come from inner
    cross-fitting, so the Cox model never sees in-sample (over-optimistic) detector output.
    """

    FEATURES: ClassVar[dict[str, list[str]]] = {"ratio": ["ratio"], "ratio_full": ["ratio", "n_pos", "max_s", "log_n"], "burden": ["n_pos", "max_s", "log_n"]}

    def __init__(self, features: str = "ratio_full", n_components: int = 32, l2: float = 1e-2, inner_folds: int = 5,
                 alpha: float = 1.0, steps: int = 300) -> None:
        self.features, self.n_components, self.l2, self.inner_folds, self.alpha, self.steps = features, n_components, l2, inner_folds, alpha, steps

    # -- node detector
    def _nodes(self, c, idx):
        z = np.concatenate([c.bags[i] for i in idx]); owner = np.repeat(np.arange(len(idx)), [len(c.bags[i]) for i in idx])
        return z, owner

    def _fit_detector(self, c, idx):
        z, owner = self._nodes(c, idx); y = c.node_positive[idx]; known = ~np.isnan(y)
        zt = torch.as_tensor(self.pca.transform((z - self.mean) / self.std), dtype=torch.float32)
        own, lab = torch.as_tensor(owner), torch.as_tensor(np.nan_to_num(y), dtype=torch.float32); keep = torch.as_tensor(known)
        w = torch.zeros(zt.shape[1], requires_grad=True); b = torch.full((1,), -4.0, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.05)
        for _ in range(self.steps):
            opt.zero_grad()
            h = torch.nn.functional.softplus(zt @ w + b)
            total = torch.zeros(len(idx)).index_add_(0, own, h)                       # sum of hazards per patient
            p = (1 - torch.exp(-total)).clamp(1e-5, 1 - 1e-5)
            loss = torch.nn.functional.binary_cross_entropy(p[keep], lab[keep]) + self.l2 * (w ** 2).sum()
            loss.backward(); opt.step()
        return w.detach(), b.detach()

    def _scores(self, c, idx, detector):
        z, owner = self._nodes(c, idx)
        zt = torch.as_tensor(self.pca.transform((z - self.mean) / self.std), dtype=torch.float32)
        s = (1 - torch.exp(-torch.nn.functional.softplus(zt @ detector[0] + detector[1]))).numpy()
        return [s[owner == k] for k in range(len(idx))]

    def _patient_features(self, scores):
        rows = []
        for s in scores:
            rows.append({"n_pos": s.sum(), "ratio": s.sum() / len(s), "max_s": s.max(), "log_n": np.log(len(s))})
        return rows

    def _matrix(self, rows):
        return np.array([[r[k] for k in self.FEATURES[self.features]] for r in rows])

    def fit(self, c, idx):
        allnodes = np.concatenate([c.bags[i] for i in idx])
        self.mean, self.std = allnodes.mean(0), np.where(allnodes.std(0) < 1e-8, 1.0, allnodes.std(0))
        self.pca = PCA(self.n_components, random_state=0).fit((allnodes - self.mean) / self.std)
        # inner cross-fitting: out-of-fold node scores for the training patients
        from sklearn.model_selection import StratifiedKFold
        y = np.nan_to_num(c.node_positive[idx], nan=0.0).astype(int); oof = [None] * len(idx)
        for tr, te in StratifiedKFold(self.inner_folds, shuffle=True, random_state=0).split(np.zeros(len(idx)), y):
            det = self._fit_detector(c, idx[tr])
            for k, s in zip(te, self._scores(c, idx[te], det)): oof[k] = s
        self.detector = self._fit_detector(c, idx)
        x = np.column_stack([c.age[idx], c.male[idx], self._matrix(self._patient_features(oof))])
        self.xm, self.xs = x.mean(0), np.where(x.std(0) < 1e-8, 1.0, x.std(0))
        self.model = CoxPHSurvivalAnalysis(alpha=self.alpha).fit((x - self.xm) / self.xs, c.y[idx])
        return self

    def risk(self, c, idx):
        rows = self._patient_features(self._scores(c, idx, self.detector))
        x = np.column_stack([c.age[idx], c.male[idx], self._matrix(rows)])
        return self.model.predict((x - self.xm) / self.xs)

    def extra_metrics(self, c, idx):
        """Does the image-derived ratio match the pathologist's? (AUC for any positive node, Spearman for the ratio)"""
        from scipy.stats import spearmanr
        from sklearn.metrics import roc_auc_score
        rows = self._patient_features(self._scores(c, idx, self.detector)); out = {}
        y = c.node_positive[idx]; ok = ~np.isnan(y)
        if ok.sum() > 3 and 0 < y[ok].sum() < ok.sum():
            out["node_pos_auc"] = roc_auc_score(y[ok], [r["max_s"] for r, o in zip(rows, ok) if o])
        ratio = c.node_ratio[idx]; ok = ~np.isnan(ratio) & (ratio > 0)
        if ok.sum() > 5:
            out["ratio_spearman_pos"] = spearmanr(ratio[ok], [r["ratio"] for r, o in zip(rows, ok) if o])[0]
        return out


class TileRatioCox:
    """Tile-level version of ``NodeRatioCox``: how much tumour does each node contain?

    A linear detector scores every tile; a patient is scored by the mean of its ``k`` highest tile logits, so a few tumour tiles
    among thousands of benign ones decide, and it is trained with the patient-level "has a positive node" label only. A node counts
    as positive when at least ``min_tiles`` of its tiles are tumour-like (about 0.13 mm2 for 8 tiles), which gives the lymph-node
    ratio, the tumour area and the largest tumour fraction of a node. They go into a ridge Cox model with age and sex; scores of the
    training patients come from inner cross-fitting.
    """

    TILE_MM2 = 0.0164
    FEATURES: ClassVar[dict[str, list[str]]] = {
        "tile_ratio": ["ratio", "topk"],
        "tile_burden": ["n_pos", "log_area", "topk", "log_n"],
        "tile_full": ["ratio", "n_pos", "log_area", "topk", "max_frac", "log_n"],
    }

    def __init__(self, features: str = "tile_full", k: int = 64, l2: float = 1e-2, steps: int = 150, refresh: int = 10,
                 inner_folds: int = 3, alpha: float = 1.0, min_tiles: int = 8) -> None:
        self.features, self.k, self.l2, self.steps, self.refresh = features, k, l2, steps, refresh
        self.inner_folds, self.alpha, self.min_tiles = inner_folds, alpha, min_tiles

    def _tile_logits(self, c, patient, w, b):
        s, e = c.tile_range[patient]
        return c.tiles.X[s:e].float() @ w + b

    def _fit_detector(self, c, idx):
        dev = c.tiles.device; y = c.node_positive[idx]; keep = np.flatnonzero(~np.isnan(y))
        patients = [int(idx[i]) for i in keep]; target = torch.as_tensor(y[keep], dtype=torch.float32, device=dev)
        d = c.tiles.X.shape[1]; w = torch.zeros(d, device=dev, requires_grad=True); b = torch.zeros(1, device=dev, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.05)
        sel = None
        for step in range(self.steps):
            if step % self.refresh == 0:   # the k highest tiles of every patient under the current detector
                with torch.no_grad():
                    sel = [c.tile_range[p][0] + self._tile_logits(c, p, w, b).topk(min(self.k, c.tile_range[p][1] - c.tile_range[p][0])).indices
                           for p in patients]
            owner = torch.cat([torch.full((len(t),), j, device=dev) for j, t in enumerate(sel)]); flat = torch.cat(sel)
            vals = c.tiles.X[flat].float() @ w + b
            logit = torch.zeros(len(patients), device=dev).index_add_(0, owner, vals) / torch.bincount(owner, minlength=len(patients))
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target) + self.l2 * (w ** 2).sum()
            loss.backward(); opt.step()
        return w.detach(), b.detach()

    @torch.no_grad()
    def _patient_features(self, c, patient, detector):
        w, b = detector; s, e = c.tile_range[patient]
        logit = self._tile_logits(c, patient, w, b)
        topk = logit.topk(min(self.k, len(logit))).values.mean().item()
        soft, hard = torch.sigmoid(logit).cpu().numpy(), (logit > 0).cpu().numpy()
        _, inv = np.unique(c.tiles.node[s:e], return_inverse=True)
        tiles = np.bincount(inv); node_hard = np.bincount(inv, weights=hard); node_soft = np.bincount(inv, weights=soft)
        n_pos = int((node_hard >= self.min_tiles).sum())
        return {"topk": topk, "n_pos": n_pos, "ratio": n_pos / len(tiles), "log_area": np.log1p(soft.sum() * self.TILE_MM2),
                "max_frac": float((node_soft / tiles).max()), "log_n": np.log(len(tiles))}

    def _matrix(self, rows):
        return np.array([[r[k] for k in self.FEATURES[self.features]] for r in rows])

    def fit(self, c, idx):
        from sklearn.model_selection import StratifiedKFold
        y = np.nan_to_num(c.node_positive[idx], nan=0.0).astype(int); oof = [None] * len(idx)
        for tr, te in StratifiedKFold(self.inner_folds, shuffle=True, random_state=0).split(np.zeros(len(idx)), y):
            det = self._fit_detector(c, idx[tr])
            for k in te: oof[k] = self._patient_features(c, int(idx[k]), det)
        self.detector = self._fit_detector(c, idx)
        x = np.column_stack([c.age[idx], c.male[idx], self._matrix(oof)])
        self.xm, self.xs = x.mean(0), np.where(x.std(0) < 1e-8, 1.0, x.std(0))
        self.model = CoxPHSurvivalAnalysis(alpha=self.alpha).fit((x - self.xm) / self.xs, c.y[idx])
        return self

    def _rows(self, c, idx):
        return [self._patient_features(c, int(i), self.detector) for i in idx]

    def risk(self, c, idx):
        x = np.column_stack([c.age[idx], c.male[idx], self._matrix(self._rows(c, idx))])
        return self.model.predict((x - self.xm) / self.xs)

    def extra_metrics(self, c, idx):
        from scipy.stats import spearmanr
        from sklearn.metrics import roc_auc_score
        rows = self._rows(c, idx); out = {}
        y = c.node_positive[idx]; ok = ~np.isnan(y)
        if ok.sum() > 3 and 0 < y[ok].sum() < ok.sum():
            out["node_pos_auc"] = roc_auc_score(y[ok], [r["topk"] for r, o in zip(rows, ok) if o])
        ratio = c.node_ratio[idx]; ok = ~np.isnan(ratio) & (ratio > 0)
        if ok.sum() > 5:
            out["ratio_spearman_pos"] = spearmanr(ratio[ok], [r["ratio"] for r, o in zip(rows, ok) if o])[0]
        return out


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

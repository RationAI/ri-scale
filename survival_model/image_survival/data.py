import numpy as np
import pandas as pd
from sksurv.util import Surv

from survival_model.features import derive_features


class TileStore:
    """Projected tiles of all patients (``build_tiles.py``); tiles of patient i are ``X[offsets[i]:offsets[i + 1]]``."""

    def __init__(self, path: str, device: str | None = None) -> None:
        import torch
        z = np.load(path, allow_pickle=False)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.X = torch.as_tensor(z["X"]).to(self.device)            # float16
        self.node, self.offsets = z["node"], z["offsets"]
        self.index = {p: i for i, p in enumerate(z["patients"])}

    def range(self, patient_id: str) -> tuple[int, int]:
        i = self.index[patient_id]
        return int(self.offsets[i]), int(self.offsets[i + 1])


class Cohort:
    """Patients with their node bags. ``bags[i]`` are the node embeddings of patient i."""

    def __init__(self, nodes_path: str, patients_path: str, embedding: str = "core", min_nodes: int = 1,
                 tiles_path: str | None = None) -> None:
        """``embedding``: "core" (node tiles) or "both" (core | margin, a node without margin tiles repeats its core)."""
        patients = pd.read_csv(patients_path) if patients_path.endswith(".csv") else pd.read_parquet(patients_path)
        z = np.load(nodes_path, allow_pickle=False)
        emb = z["core"]
        if embedding == "both":
            emb = np.concatenate([emb, np.where(np.isnan(z["margin"]), z["core"], z["margin"])], axis=1)
        self.node_table = pd.DataFrame({k: z[k] for k in ("patient", "slide_idx", "node", "n_core", "n_margin", "area_mm2")})
        keep = []
        bags, nodes = [], []
        for pid in patients["patient_id"]:
            idx = np.flatnonzero(z["patient"] == pid)
            if len(idx) >= min_nodes:
                keep.append(pid); bags.append(emb[idx].astype(np.float32)); nodes.append(self.node_table.iloc[idx])
        self.patients = patients.set_index("patient_id").loc[keep].reset_index()
        self.bags, self.nodes = bags, nodes
        self.tiles = TileStore(tiles_path) if tiles_path else None
        self.tile_range = [self.tiles.range(p) for p in self.patients["patient_id"]] if self.tiles else None
        # Excel-derived features (stage, M, markers) exist only when the full patient table is given
        self.features = derive_features(self.patients) if "nodes_positive" in self.patients else None
        # Pathologist's node counts: training labels only (weak supervision of the node detector), never model inputs
        self.node_positive, self.node_ratio = self._node_labels()
        self.age = self.patients["age"].to_numpy(float)
        self.male = (self.patients["sex"] == "M").to_numpy(float)
        self.y = Surv.from_arrays(self.patients["os_event"].astype(bool).to_numpy(),
                                  self.patients["time_days"].to_numpy(float) / 365.25)

    def _node_labels(self):
        """1 / 0 / nan: does the patient have a positive node; and positive / examined nodes (nan when unknown)."""
        pos = np.full(len(self.patients), np.nan); ratio = np.full(len(self.patients), np.nan)
        if "nodes_positive" in self.patients:
            n_pos, n_exam = self.patients["nodes_positive"].to_numpy(float), self.patients["nodes_examined"].to_numpy(float)
            pos = np.where(np.isnan(n_pos), np.nan, (n_pos > 0).astype(float))
            ratio = np.where((n_exam > 0) & ~np.isnan(n_pos), n_pos / np.where(n_exam > 0, n_exam, 1), np.nan)
            if self.features is not None:  # no counts: fall back on the reported pN
                pn = self.features["pn_group"].map({"N0": 0.0, "N1": 1.0, "N2": 1.0}).to_numpy(float)
                pos = np.where(np.isnan(pos), pn, pos)
        return pos, ratio

    def __len__(self) -> int:
        return len(self.bags)

    def summary(self) -> dict:
        n_nodes = np.array([len(b) for b in self.bags])
        return {"patients": len(self), "events": int(self.y["event"].sum()), "nodes_total": int(n_nodes.sum()),
                "nodes_per_patient": [int(n_nodes.min()), float(np.median(n_nodes)), int(n_nodes.max())]}

    def hand_features(self) -> np.ndarray:
        """Cheap per-patient numbers: log node count, log total / mean / max node area, mean log tiles per node."""
        out = []
        for nodes in self.nodes:
            area = nodes["area_mm2"].to_numpy()
            out.append([np.log(len(nodes)), np.log1p(area.sum()), np.log1p(area.mean()), np.log1p(area.max()),
                        np.log1p(nodes["n_core"].to_numpy()).mean()])
        return np.asarray(out)

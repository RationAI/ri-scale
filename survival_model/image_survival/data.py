import numpy as np
import pandas as pd
from sksurv.util import Surv


class Cohort:
    """Patients with their node bags. ``bags[i]`` are the node embeddings of patient i."""

    def __init__(self, nodes_path: str, patients_path: str, margin: bool = False, min_nodes: int = 1) -> None:
        patients = pd.read_csv(patients_path) if patients_path.endswith(".csv") else pd.read_parquet(patients_path)
        z = np.load(nodes_path, allow_pickle=False)
        emb = z["margin"] if margin else z["core"]
        self.node_table = pd.DataFrame({k: z[k] for k in ("patient", "slide_idx", "node", "n_core", "n_margin", "area_mm2")})
        keep = []
        bags, nodes = [], []
        for pid in patients["patient_id"]:
            idx = np.flatnonzero(z["patient"] == pid)
            if len(idx) >= min_nodes:
                keep.append(pid); bags.append(emb[idx].astype(np.float32)); nodes.append(self.node_table.iloc[idx])
        self.patients = patients.set_index("patient_id").loc[keep].reset_index()
        self.bags, self.nodes = bags, nodes
        self.age = self.patients["age"].to_numpy(float)
        self.male = (self.patients["sex"] == "M").to_numpy(float)
        self.y = Surv.from_arrays(self.patients["os_event"].astype(bool).to_numpy(),
                                  self.patients["time_days"].to_numpy(float) / 365.25)

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

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sksurv.util import Surv

log = logging.getLogger(__name__)

UNKNOWN = "unknown"
OTHER = "other"


# ── raw features ───────────────────────────────────────────────────────────────

_LOCATIONS = {
    "right": ("C18.0", "C18.1", "C18.2", "C18.3", "C18.4"),  # caecum .. transverse colon
    "left": ("C18.5", "C18.6", "C18.7"),                     # splenic flexure .. sigmoid
    "rectum": ("C19", "C20"),
}
_MORPHOLOGY_GROUPS = {
    "mucinous": ("8480", "8481"),
    "signet_ring": ("8490",),
    "adenocarcinoma": ("8140", "8210", "8211", "8213", "8220", "8221", "8261", "8262", "8263"),
}
_GRADE_GROUPS = {"G1": "low", "G2": "low", "low": "low", "G3": "high", "G4": "high", "high": "high"}
_STAGE_GROUPS = {"0": "0-I", "I": "0-I", "II": "II", "III": "III", "IV": "IV"}


def _first_match(code: Any, groups: dict[str, tuple[str, ...]], default: str | None) -> str | None:
    if not isinstance(code, str):
        return None
    return next((name for name, prefixes in groups.items() if code.startswith(prefixes)), default)


def _pt_group(pt: Any) -> str | None:
    if not isinstance(pt, str):
        return None
    return "T0-2" if pt in ("T0", "Tis", "T1", "T2") else pt[:2]


def _prefix(value: Any, length: int = 2) -> str | None:
    return value[:length] if isinstance(value, str) else None


def derive_features(patients: pd.DataFrame) -> pd.DataFrame:
    """All candidate model features; the features config picks which ones are used."""
    site = patients["topography"].fillna(patients["diagnosis"])
    examined = patients["nodes_examined"].astype(float)
    return pd.DataFrame(
        {
            "age": patients["age"].astype(float),
            "diagnosis_year": patients["diagnosis_year"].astype(float),
            "nodes_examined": examined,
            "nodes_positive": patients["nodes_positive"].astype(float),
            "ln_ratio": patients["nodes_positive"] / examined.where(examined > 0),
            "sex": patients["sex"],
            # C18.8 / C18.9 (overlapping / NOS) stay unknown
            "location": site.map(lambda c: _first_match(c, _LOCATIONS, None)),
            "morphology_group": patients["morphology"].map(
                lambda c: _first_match(c, _MORPHOLOGY_GROUPS, OTHER)
            ),
            "grade_group": patients["grade"].map(_GRADE_GROUPS),
            "pt_group": patients["pt"].map(_pt_group),
            "pn_group": patients["pn"].map(_prefix),
            "m_group": patients["m"].map(_prefix),
            "stage_group": patients["stage_group"].map(_STAGE_GROUPS),
            "neoadjuvant": patients["neoadjuvant"].map({True: "yes", False: "no"}),
            "mmr": patients["mmr"],
            "kras": patients["kras"],
            "nras": patients["nras"],
            "braf": patients["braf"],
        },
        index=patients.index,
    )


def make_target(patients: pd.DataFrame, endpoint: str) -> np.ndarray:
    """Structured (event, time) array; time in years since diagnosis."""
    return Surv.from_arrays(
        event=patients[f"{endpoint}_event"].astype(bool).to_numpy(),
        time=patients["time_days"].to_numpy(dtype=float) / 365.25,
    )


# ── encoding ───────────────────────────────────────────────────────────────────

class TabularEncoder(TransformerMixin, BaseEstimator):
    """Median imputation with missing indicators for numeric features, dummy coding for categorical ones.

    Missing categories form their own "unknown" level and levels rarer than
    ``min_category_count`` are merged into "other" (or into the reference level
    when "other" would still be too rare). Everything is learnt in ``fit``, so the
    encoder can sit inside a pipeline that is refitted per CV fold.

    Encoded columns identical to an earlier one on the training data (e.g.
    ``pt_group=unknown`` and ``pn_group=unknown`` for patients without resection)
    are dropped, categorical columns being kept in preference to numeric ones.

    ``feature_groups_`` maps every raw feature to its encoded columns, so a feature
    can be tested or dropped as a whole.
    """

    def __init__(
        self,
        numeric: Sequence[str] = (),
        categorical: Sequence[str] = (),
        reference: dict[str, str] | None = None,
        min_category_count: int = 10,
        scale_numeric: bool = True,
    ) -> None:
        self.numeric = numeric
        self.categorical = categorical
        self.reference = reference
        self.min_category_count = min_category_count
        self.scale_numeric = scale_numeric

    @staticmethod
    def _categories(values: pd.Series) -> pd.Series:
        return values.astype(object).where(values.notna(), UNKNOWN).astype(str)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TabularEncoder":
        self.numeric_params_: dict[str, tuple[float, float, float]] = {}
        self.levels_: dict[str, tuple[set[str], str, list[str]]] = {}
        self.feature_groups_: dict[str, list[str]] = {}

        for feature in self.numeric:
            values = X[feature].astype(float)
            if values.notna().sum() == 0:
                log.warning("Numeric feature %r has no values, skipping it", feature)
                self.feature_groups_[feature] = []
                continue
            median = float(values.median())
            filled = values.fillna(median)
            mean, std = (float(filled.mean()), float(filled.std()) or 1.0) if self.scale_numeric else (0.0, 1.0)
            self.numeric_params_[feature] = (median, mean, std)
            self.feature_groups_[feature] = [feature] + ([f"{feature}_missing"] if values.isna().any() else [])

        for feature in self.categorical:
            counts = self._categories(X[feature]).value_counts()
            rare = counts[counts < self.min_category_count]
            counts = counts.drop(rare.index)
            if rare.sum() >= self.min_category_count:
                counts[OTHER] = counts.get(OTHER, 0) + rare.sum()
            if counts.empty:
                log.warning("Categorical feature %r has no frequent level, skipping it", feature)
                self.feature_groups_[feature] = []
                continue

            reference = (self.reference or {}).get(feature)
            if reference not in counts.index:
                reference = str(counts.idxmax())
            dummies = sorted(level for level in counts.index if level != reference)
            self.levels_[feature] = (set(counts.index), reference, dummies)
            self.feature_groups_[feature] = [f"{feature}={level}" for level in dummies]
            if not dummies:
                log.warning("Categorical feature %r is constant (%s)", feature, reference)

        encoded = self._encode(X)
        preference = [c for f in self.categorical for c in self.feature_groups_[f]]
        preference += [c for c in encoded.columns if c not in preference]
        duplicated = set(encoded[preference].columns[encoded[preference].T.duplicated()])
        if duplicated:
            log.info("Dropping encoded columns identical to an earlier one: %s", sorted(duplicated))
        self.feature_groups_ = {
            feature: [c for c in columns if c not in duplicated]
            for feature, columns in self.feature_groups_.items()
        }
        self.feature_names_out_ = [c for columns in self.feature_groups_.values() for c in columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._encode(X)[self.feature_names_out_]

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, pd.Series] = {}
        for feature, (median, mean, std) in self.numeric_params_.items():
            values = X[feature].astype(float)
            out[feature] = (values.fillna(median) - mean) / std
            if f"{feature}_missing" in self.feature_groups_[feature]:
                out[f"{feature}_missing"] = values.isna().astype(float)

        for feature, (levels, reference, dummies) in self.levels_.items():
            categories = self._categories(X[feature])
            unseen = OTHER if OTHER in levels else reference
            categories = categories.where(categories.isin(levels), unseen)
            for level in dummies:
                out[f"{feature}={level}"] = (categories == level).astype(float)

        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)

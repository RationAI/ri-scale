import itertools
import logging
import re
from pathlib import Path

import pandas as pd

from preprocessing.value_parsers import normalize_header

log = logging.getLogger(__name__)

ColumnPatterns = dict[str, list[str]]


def match_columns(
    headers: list[str], columns: ColumnPatterns, strict: bool = True
) -> dict[int, str]:
    """Map header positions to canonical column names.

    Canonical columns are resolved in config order and each header is used at
    most once. Canonical columns with identical patterns (e.g. two "sn_uzliny"
    columns whose full names are unknown) take the matching headers in sheet
    order. With ``strict``, more matching headers than canonical columns is an
    error, so a too loose pattern never silently picks the wrong column.
    """
    groups: dict[tuple[str, ...], list[str]] = {}
    for canonical, patterns in columns.items():
        groups.setdefault(tuple(patterns), []).append(canonical)

    normalized = [normalize_header(header) for header in headers]
    mapping: dict[int, str] = {}
    for patterns, canonicals in groups.items():
        hits = [
            i
            for i, header in enumerate(normalized)
            if i not in mapping and header and any(re.search(p, header) for p in patterns)
        ]
        if strict and len(hits) > len(canonicals):
            raise ValueError(
                f"Columns {canonicals} match {len(hits)} headers: {[headers[i] for i in hits]}; "
                "make their patterns in dataset.columns more specific"
            )
        mapping |= dict(zip(hits, canonicals, strict=False))
    return mapping


class ExcelDataSource:
    """Rows of every sheet of every matching Excel file, with canonical column names.

    The header row is searched for in the first ``header_scan_rows`` rows (exports
    often start with a title line); sheets without a recognisable header, such as
    legends, are skipped.
    """

    def __init__(
        self,
        src_dir: str,
        glob_pattern: str | list[str] = "*.xlsx",
        sheets: str | int | list[str | int] | None = None,
        header_scan_rows: int = 20,
        min_header_matches: int = 5,
    ) -> None:
        patterns = [glob_pattern] if isinstance(glob_pattern, str) else list(glob_pattern)
        self.paths = sorted(
            set(itertools.chain(*(Path(src_dir).rglob(p) for p in patterns)))
        )
        # Skip Excel lock files of documents that are open
        self.paths = [p for p in self.paths if not p.name.startswith("~$")]
        self.sheets = sheets
        self.header_scan_rows = header_scan_rows
        self.min_header_matches = min_header_matches

    def read(self, columns: ColumnPatterns) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
        """Returns the concatenated rows and, per sheet, the canonical -> header mapping."""
        if not self.paths:
            raise ValueError("No Excel files found")

        frames: list[pd.DataFrame] = []
        mappings: dict[str, dict[str, str]] = {}
        for path in self.paths:
            sheets = pd.read_excel(path, sheet_name=self.sheets, header=None, dtype=object)
            if not isinstance(sheets, dict):
                sheets = {self.sheets: sheets}
            for sheet_name, raw in sheets.items():
                source = f"{path.name}::{sheet_name}"
                frame, mapping = self._read_sheet(raw, columns, source)
                if frame is not None:
                    frames.append(frame)
                    mappings[source] = mapping

        if not frames:
            raise ValueError("No sheet with a recognisable header found")
        return pd.concat(frames, ignore_index=True), mappings

    def _read_sheet(
        self, raw: pd.DataFrame, columns: ColumnPatterns, source: str
    ) -> tuple[pd.DataFrame | None, dict[str, str]]:
        best_row, best_mapping = None, {}
        for row in range(min(self.header_scan_rows, len(raw))):
            headers = ["" if pd.isna(h) else str(h) for h in raw.iloc[row]]
            mapping = match_columns(headers, columns, strict=False)
            if len(mapping) > len(best_mapping):
                best_row, best_mapping = row, mapping

        if best_row is None or len(best_mapping) < self.min_header_matches:
            log.warning("Skipping %s: no header row found", source)
            return None, {}

        headers = ["" if pd.isna(h) else str(h) for h in raw.iloc[best_row]]
        best_mapping = match_columns(headers, columns)
        positions = list(best_mapping)
        frame = raw.iloc[best_row + 1 :, positions].copy()
        frame.columns = [best_mapping[i] for i in positions]
        frame = frame.dropna(how="all").reset_index(drop=True)
        frame["source"] = source

        log.info("%s: %d rows, header on row %d", source, len(frame), best_row + 1)
        return frame, {best_mapping[i]: headers[i] for i in positions}

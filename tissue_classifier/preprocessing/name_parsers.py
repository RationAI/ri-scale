import re
from dataclasses import dataclass
from pathlib import Path

LN_LABELS = frozenset({"LNN", "LNP"})
COLORECTUM_LABELS = frozenset({"CA", "T", "N"})
_ALL_KNOWN_LABELS = frozenset({"LNN", "LNP", "CA", "T", "N", "MET", "M", "GMN"})

_LABEL_ALT = "|".join(sorted(_ALL_KNOWN_LABELS, key=len, reverse=True))
# Accepts both underscore and hyphen as year-case separator: 2016_01360 or 2016-01360
_PATTERN = re.compile(
    r"^(\d{4})([-_])(\d+)-(\d+)-(" + _LABEL_ALT + r")\.mrxs$"
)


@dataclass
class ParsedFilename:
    case_id: str       # e.g. "2016_01360" or "2016-01360" (preserves original separator)
    slide_num: str     # e.g. "02"
    label: str         # raw suffix, e.g. "LNN"
    tissue_type: str   # "LN" | "colorectum" | "excluded"
    excluded: bool


def parse_filename(filename: str) -> ParsedFilename:
    name = Path(filename).name
    match = _PATTERN.match(name)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename!r}")

    year, sep, case_num, slide_num, label = match.groups()
    case_id = f"{year}_{case_num}"

    if label in LN_LABELS:
        tissue_type = "LN"
        excluded = False
    elif label in COLORECTUM_LABELS:
        tissue_type = "colorectum"
        excluded = False
    else:
        tissue_type = "excluded"
        excluded = True

    return ParsedFilename(
        case_id=case_id,
        slide_num=slide_num,
        label=label,
        tissue_type=tissue_type,
        excluded=excluded,
    )

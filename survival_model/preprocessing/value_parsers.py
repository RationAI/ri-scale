"""Parsers for single cells of the clinical export.

Every parser takes one non-missing raw cell and returns its canonical value,
``None`` when the cell explicitly says "unknown" (TX, GX, "nevyšetřeno", ...),
or raises ``ValueError`` when the value is not understood. Unparsed values are
counted in the parse report so the rules here can be extended.
"""

import re
import unicodedata
from datetime import date, datetime
from typing import Any

import pandas as pd

_MISSING = frozenset({"", "-", "--", "?", "nan", "nat", "none", "null", "n/a"})
_EXCEL_EPOCH = pd.Timestamp("1899-12-30")


# ── generic helpers ────────────────────────────────────────────────────────────

def is_missing(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _MISSING
    return bool(pd.isna(value))


def as_text(value: Any) -> str:
    """String form of a cell; integral floats lose their ".0" (Excel stores 1 as 1.0)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def normalize_text(value: Any) -> str:
    """Lower-case, strip diacritics and collapse whitespace."""
    text = unicodedata.normalize("NFKD", as_text(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.lower().split())


def normalize_header(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", normalize_text(value)).strip("_")


def _match_keywords(text: str, rules: list[tuple[str | None, tuple[str, ...]]]) -> str | None:
    """Return the label of the first rule with a keyword contained in ``text``."""
    for label, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return label
    raise ValueError(text)


# ── identifiers ────────────────────────────────────────────────────────────────

_YEAR_FIRST = re.compile(r"^(\d{4})\D+(\d+)$")
_YEAR_LAST = re.compile(r"^(\d+)\D+(\d{2}|\d{4})$")


def normalize_case_id(value: Any) -> str:
    """Bring a pathology case number to the ``YYYY_NNNNN`` form used in slide names.

    Accepts e.g. "2016/01360", "2016-1360", "1360/16", "B 01360/2016". Unknown
    formats are returned unchanged.
    """
    raw = as_text(value)
    text = re.sub(r"^[A-Za-z]+\s*", "", raw)
    for pattern, year_group in ((_YEAR_FIRST, 1), (_YEAR_LAST, 2)):
        match = pattern.match(text)
        if not match:
            continue
        year, number = match.group(year_group), match.group(3 - year_group)
        year_int = int(year) + (2000 if len(year) == 2 else 0)
        if 1990 <= year_int <= 2100:
            return f"{year_int}_{int(number):05d}"
    return raw


# ── dates and numbers ──────────────────────────────────────────────────────────

_ISO_DATE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})")
_CZ_DATE = re.compile(r"^(\d{1,2})[./]\s*(\d{1,2})[./]\s*(\d{4})")


def parse_date(value: Any) -> pd.Timestamp:
    """Excel date cells, Excel serial numbers, ISO and Czech (d.m.yyyy) strings.

    A bare year (anonymised birth dates) is placed in the middle of that year.
    """
    if isinstance(value, datetime | date):
        return pd.Timestamp(value)
    if isinstance(value, int | float):
        if 1900 <= value <= 2100:
            return pd.Timestamp(int(value), 7, 1)
        if 0 < value < 80_000:
            return _EXCEL_EPOCH + pd.Timedelta(days=float(value))
        raise ValueError(value)

    text = as_text(value)
    if match := _ISO_DATE.match(text):
        year, month, day = match.groups()
    elif match := _CZ_DATE.match(text):
        day, month, year = match.groups()
    elif re.fullmatch(r"\d{4}", text):
        return pd.Timestamp(int(text), 7, 1)
    else:
        raise ValueError(text)
    return pd.Timestamp(int(year), int(month), int(day))


def parse_count(value: Any) -> float:
    count = float(value) if isinstance(value, int | float) else float(as_text(value).replace(",", "."))
    if not 0 <= count <= 200:
        raise ValueError(value)
    return count


_TRUE = frozenset({"y", "r", "1", "a", "ano", "yes", "true", "x", "+"})
_FALSE = frozenset({"0", "n", "ne", "no", "false"})


def parse_flag(value: Any) -> bool:
    text = normalize_text(value)
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(text)


# ── demographics ───────────────────────────────────────────────────────────────

# 1 / 2 are the ÚZIS registry codes
_SEX = {"m": "M", "muz": "M", "male": "M", "1": "M", "z": "F", "zena": "F", "f": "F", "female": "F", "2": "F"}


def parse_sex(value: Any) -> str:
    text = normalize_text(value)
    if text not in _SEX:
        raise ValueError(text)
    return _SEX[text]


# ── tumour classification ──────────────────────────────────────────────────────

_ICD = re.compile(r"([A-Z])\s*(\d{2})(?:\s*[.,]?\s*(\d))?")


def parse_icd(value: Any) -> str:
    """First ICD-10 / ICD-O-3 topography code in the cell, e.g. "C187 sigma" -> "C18.7"."""
    match = _ICD.search(as_text(value).upper())
    if not match:
        raise ValueError(value)
    letter, category, subcategory = match.groups()
    return f"{letter}{category}.{subcategory}" if subcategory else f"{letter}{category}"


_MORPHOLOGY = re.compile(r"(\d{4})\s*/?\s*(\d)?")


def parse_morphology(value: Any) -> str:
    """ICD-O-3 morphology, e.g. "81403" or "M-8140/3" -> "8140/3"."""
    match = _MORPHOLOGY.search(as_text(value))
    if not match:
        raise ValueError(value)
    code, behaviour = match.groups()
    return f"{code}/{behaviour}" if behaviour else code


_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4"}


def parse_grade(value: Any) -> str | None:
    text = normalize_text(value).replace(" ", "")
    if text in {"x", "gx", "9", "g9"}:
        return None
    if match := re.fullmatch(r"g?([1-4]|iv|i{1,3})", text):
        return "G" + _ROMAN.get(match.group(1), match.group(1))
    return _match_keywords(text, [("low", ("low", "nizk", "lg")), ("high", ("high", "vysok", "hg"))])


_TNM = {
    category: re.compile(rf"^([ycrpau]*){category.lower()}?(is|x|[0-4][a-d]?)")
    for category in "TNM"
}


def _tnm_match(value: Any, category: str) -> re.Match[str]:
    match = _TNM[category].match(normalize_text(value).replace(" ", ""))
    if not match:
        raise ValueError(value)
    return match


def parse_tnm(value: Any, category: str) -> str | None:
    """"ypT4a" -> "T4a", "3" -> "T3", "pNX" -> None (unknown)."""
    code = _tnm_match(value, category).group(2)
    if code == "x":
        return None
    return category + (code if code == "is" else code.upper()[0] + code[1:])


def tnm_prefix(value: Any, category: str) -> str:
    """The y/r/p/c... prefix letters written in front of a TNM value ("" if none)."""
    try:
        return _tnm_match(value, category).group(1)
    except ValueError:
        return ""


def parse_t(value: Any) -> str | None:
    return parse_tnm(value, "T")


def parse_n(value: Any) -> str | None:
    return parse_tnm(value, "N")


def parse_m(value: Any) -> str | None:
    return parse_tnm(value, "M")


_STAGE = re.compile(r"^(?:stadium|stage|st\.?)?\s*(iv|i{1,3}|[0-4])\s*([abc])?$")
_STAGE_ROMAN = {"0": "0", "1": "I", "2": "II", "3": "III", "4": "IV"}


def parse_stage(value: Any) -> str | None:
    """UICC stage, e.g. "IIIb" / "3B" / "stadium 3" -> "IIIB" / "IIIB" / "III"."""
    text = normalize_text(value)
    if text in {"x", "9", "99", "nezname", "neurceno"}:
        return None
    match = _STAGE.match(text)
    if not match:
        raise ValueError(text)
    main, sub = match.groups()
    return _STAGE_ROMAN.get(main, main.upper()) + (sub.upper() if sub else "")


# ── molecular markers ──────────────────────────────────────────────────────────

_NOT_TESTED = ("nevys", "neprov", "nehodnot", "nelze", "neurc", "nezn", "not tested", "n.a.")

# Order matters: "nemutovaný" and "bez mutace" contain "mut".
_MUTATION_RULES: list[tuple[str | None, tuple[str, ...]]] = [
    (None, _NOT_TESTED),
    ("wt", ("wt", "wild", "divok", "nemut", "bez mut", "nebyla", "nezjist", "neprok", "negat", "neg")),
    ("mut", ("mut", "pozit", "pos", "g12", "g13", "q61", "a146", "k117", "v600", "exon", "p.")),
]

# Loss of expression is checked before "preserved" so that mixed reports
# ("zachovalá exprese MSH2/MSH6, ztráta MLH1/PMS2") count as deficient.
_MMR_RULES: list[tuple[str | None, tuple[str, ...]]] = [
    (None, _NOT_TESTED),
    ("dMMR", ("dmmr", "msi-h", "msi h", "msih", "msi-high", "deficien", "ztrat", "loss", "chyb")),
    ("pMMR", ("pmmr", "mss", "msi-l", "msil", "stabil", "proficien", "zachov", "intakt", "preserv", "normal")),
    ("dMMR", ("msi",)),
]


def parse_mutation(value: Any) -> str | None:
    """KRAS / NRAS / BRAF status -> "wt" | "mut"."""
    return _match_keywords(normalize_text(value), _MUTATION_RULES)


def parse_mmr(value: Any) -> str | None:
    """Mismatch-repair status -> "pMMR" | "dMMR"."""
    return _match_keywords(normalize_text(value), _MMR_RULES)

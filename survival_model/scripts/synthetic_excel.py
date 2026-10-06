"""Write a synthetic clinical export with the MMCI header, for testing the pipeline without patient data.

Survival follows a known model (stage, age, grade, BRAF, MMR, ... matter; sex,
KRAS and NRAS do not), so the feature analysis can be sanity-checked. Cell formats
are deliberately mixed (C187 / C18.7, pT3 / 3, Excel dates / "d.m.yyyy", ...).

Usage:
    uv run scripts/synthetic_excel.py <out.xlsx> [--patients 1500] [--seed 0]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HEADER = [
    "id_biopsie", "id_vzorku", "topografie", "morfologie", "grading", "diagnoza", "y_prefix", "r_prefix",
    "pt", "pn", "pn_sn_uzliny_poz", "pn_sn_uzliny_vys", "pn_ost_uzliny_poz", "pn_ost_uzliny_vys",
    "pm", "histol_datum_zpracovani", "pohlavi", "datum_narozeni", "datum_umrti",
    "dgh_zakl_pric_umrti", "dgia_bezprostredna_pric_umrti", "dgic_dalsia_pric_umrti", "dgii_komorbidita",
    "navsteva_naposledy", "datum_diagnozy", "cm", "klinicke_stadium", "mmr", "kras", "nras", "braf",
]
SITES = ["C18.0", "C18.2", "C18.3", "C18.4", "C18.5", "C18.6", "C18.7", "C18.9", "C19.9", "C20.9"]
SITE_P = [0.08, 0.08, 0.04, 0.07, 0.03, 0.05, 0.25, 0.03, 0.07, 0.30]
FOLLOW_UP_END = pd.Timestamp("2025-06-30")
# Death certificate codes are written as "C187 - ZN - esovitý tračník [colon sigmoideum]"
ICD_TEXT = {
    "C18": "ZN - tlusté střevo", "C19": "Zhoubný novotvar rektosigmoideálního spojení",
    "C20": "Zhoubný novotvar konečníku", "C78": "Sekundární ZN dýchacích a trávicích orgánů",
    "I21": "Akutní infarkt myokardu", "I50": "Selhání srdce", "J18": "Pneumonie",
    "C34": "ZN - průduška a plíce", "E11": "Diabetes mellitus 2. typu", "R57": "Šok",
    "I46": "Srdeční zástava", "K72": "Selhání jater", "I10": "Esenciální hypertenze",
}


def _icd_text(code: str | None) -> str | None:
    if code is None:
        return None
    return f"{code.replace('.', '')} - {ICD_TEXT.get(code[:3], 'NS')}"
ARABIC_STAGE = {"I": "1", "II": "2", "III": "3", "IV": "4"}
DMMR_TEXTS = ["dMMR", "MSI-H", "ztráta exprese MLH1, PMS2"]
PMMR_TEXTS = ["pMMR", "MSS", "zachovalá exprese"]


def _pick(rng: np.random.Generator, options: list, p: list[float] | None = None):
    return options[rng.choice(len(options), p=p)]


def _tnm(rng: np.random.Generator, stage: str) -> tuple[str, str, str]:
    if stage == "I":
        return _pick(rng, ["T1", "T2"]), "N0", "M0"
    if stage == "II":
        return _pick(rng, ["T3", "T4a", "T4b"], [0.8, 0.15, 0.05]), "N0", "M0"
    n = _pick(rng, ["N1a", "N1b", "N1c", "N2a", "N2b"], [0.25, 0.3, 0.05, 0.25, 0.15])
    t = _pick(rng, ["T1", "T2", "T3", "T4a", "T4b"], [0.03, 0.1, 0.6, 0.2, 0.07])
    if stage == "III":
        return t, n, "M0"
    return t, _pick(rng, ["N0", n], [0.3, 0.7]), _pick(rng, ["M1a", "M1b", "M1c"], [0.6, 0.25, 0.15])


def _positive_nodes(rng: np.random.Generator, n: str) -> int:
    low, high = {"N0": (0, 0), "N1a": (1, 1), "N1b": (2, 3), "N1c": (0, 0), "N2a": (4, 6), "N2b": (7, 15)}[n]
    return int(rng.integers(low, high + 1))


def _date(rng: np.random.Generator, ts: pd.Timestamp):
    """Mostly real Excel dates, sometimes Czech-formatted text."""
    if pd.isna(ts):
        return None
    return f"{ts.day}.{ts.month}.{ts.year}" if rng.random() < 0.1 else ts.to_pydatetime()


def simulate_patient(rng: np.random.Generator, i: int) -> list[dict]:
    sex = _pick(rng, ["M", "F"], [0.58, 0.42])
    age = float(np.clip(rng.normal(67, 11), 28, 95))
    diagnosis = pd.Timestamp("2012-01-01") + pd.Timedelta(days=int(rng.integers(0, 11 * 365)))
    birth = diagnosis - pd.Timedelta(days=int(age * 365.25))
    site = _pick(rng, SITES, SITE_P)
    right, rectum = site < "C18.5", site >= "C19"
    stage = _pick(rng, ["I", "II", "III", "IV"], [0.18, 0.3, 0.32, 0.2])
    t, n, m = _tnm(rng, stage)
    operated = rng.random() < (0.5 if stage == "IV" else 0.95)
    neoadjuvant = operated and rectum and stage in ("II", "III") and rng.random() < 0.6
    morphology = _pick(rng, ["8140/3", "8480/3", "8490/3", "8210/3", "8240/3"], [0.84, 0.1, 0.02, 0.03, 0.01])
    grade = "G3" if morphology == "8490/3" else _pick(rng, ["G1", "G2", "G3"], [0.1, 0.7, 0.2])
    dmmr = rng.random() < (0.18 if right else 0.04)
    braf = rng.random() < (0.12 if right else 0.03)
    kras = not braf and rng.random() < 0.42
    nras = not (braf or kras) and rng.random() < 0.06
    tested = rng.random() < (0.9 if stage == "IV" else 0.35)

    # Cancer hazard: sex, KRAS and NRAS have no effect by construction
    lp = (
        0.035 * (age - 67)
        + {"T1": 0, "T2": 0, "T3": 0.25, "T4a": 0.7, "T4b": 0.9}[t]
        + (0.5 if n.startswith("N1") else 1.0 if n.startswith("N2") else 0.0)
        + (1.8 if m != "M0" else 0.0)
        + (0.3 if grade == "G3" else 0.0)
        + (0.6 if morphology == "8490/3" else 0.0)
        + (0.45 if braf else 0.0)
        - (0.35 if dmmr else 0.0)
        + (0.8 if not operated else 0.0)
    )
    years_cancer = (-np.log(rng.random()) / (0.035 * np.exp(lp))) ** (1 / 1.1)
    years_other = -np.log(rng.random()) / (0.015 * np.exp(0.08 * (age - 67)))
    years_lost = rng.exponential(12)
    death = diagnosis + pd.Timedelta(days=int(365.25 * min(years_cancer, years_other)))
    dead = death <= FOLLOW_UP_END and min(years_cancer, years_other) < years_lost
    if dead:
        last_visit = death - pd.Timedelta(days=int(rng.integers(0, 200)))
        cancer_death = years_cancer < years_other
        icd10_site = site[:3] if rectum else site
        underlying = (_pick(rng, [icd10_site, "C78.7"], [0.92, 0.08])
                      if cancer_death else _pick(rng, ["I21.9", "I50.9", "J18.9", "C34.9", "E11.9"]))
        immediate = _pick(rng, ["R57.9", "J18.9", "I46.9", "K72.9"])
        chain = underlying if rng.random() < 0.7 else None
    else:
        end = min(FOLLOW_UP_END, diagnosis + pd.Timedelta(days=int(365.25 * years_lost)))
        last_visit = end - pd.Timedelta(days=int(rng.integers(0, 120)))
        death, underlying, immediate, chain = pd.NaT, None, None, None
    last_visit = max(last_visit, diagnosis)

    patient = {
        "pohlavi": _pick(rng, [{"M": "M", "F": "Ž"}[sex], {"M": "1", "F": "2"}[sex]], [0.95, 0.05]),
        "datum_narozeni": _date(rng, birth),
        "datum_umrti": _date(rng, death),
        "dgh_zakl_pric_umrti": _icd_text(underlying),
        "dgia_bezprostredna_pric_umrti": "HYPOX - Popis nenájdený v číselníku"
        if dead and rng.random() < 0.02 else _icd_text(immediate),
        "dgic_dalsia_pric_umrti": _icd_text(chain),
        "dgii_komorbidita": _icd_text(_pick(rng, [None, "I10", "E11.9"])) if dead else None,
        "navsteva_naposledy": _date(rng, last_visit),
        "datum_diagnozy": _date(rng, diagnosis),
        "cm": _pick(rng, [f"c{m}", m]) if rng.random() < 0.9 else None,
        "klinicke_stadium": _pick(rng, [stage + _pick(rng, ["", "A", "B"]), ARABIC_STAGE[stage]], [0.8, 0.2])
        if rng.random() < 0.9 else None,
        "mmr": _pick(rng, DMMR_TEXTS if dmmr else PMMR_TEXTS)
        if tested else _pick(rng, [None, "nevyšetřeno"], [0.9, 0.1]),
        "kras": (_pick(rng, ["G12D", "mutace G13D", "mut"]) if kras else _pick(rng, ["wt", "divoký typ", "bez mutace"]))
        if tested else None,
        "nras": (_pick(rng, ["Q61K", "mut"]) if nras else _pick(rng, ["wt", "negativní"])) if tested else None,
        "braf": ("V600E" if braf else _pick(rng, ["wt", "negativní"])) if tested else None,
    }

    cases = []
    if not operated or rng.random() < 0.6:  # diagnostic biopsy without pathological staging
        cases.append({"date": diagnosis, "resection": False})
    if operated:
        delay = int(rng.integers(150, 210)) if neoadjuvant else int(rng.integers(5, 60))
        cases.append({"date": diagnosis + pd.Timedelta(days=delay), "resection": True})

    rows = []
    for case in cases:
        case_id = f"{case['date'].year}/{rng.integers(1, 30000):05d}"
        tumour = {
            "id_biopsie": case_id,
            "topografie": site.replace(".", "") if rng.random() < 0.5 else site,
            "morfologie": morphology.replace("/", "") if rng.random() < 0.3 else morphology,
            "grading": grade if rng.random() < 0.7 else grade[1:],
            "diagnoza": site[:3] if site >= "C19" else site,
            "histol_datum_zpracovani": _date(rng, case["date"]),
            "pn_sn_uzliny_poz": 0,  # sentinel biopsy not done -> 0, not empty
            "pn_sn_uzliny_vys": 0,
        }
        if case["resection"]:
            yp = neoadjuvant and rng.random() < 0.5  # y written in front of pT instead of y_prefix
            examined = int(rng.poisson(18))
            tumour |= {
                "y_prefix": "y" if neoadjuvant and not yp else None,
                "pt": _pick(rng, [f"{'y' if yp else ''}p{t}", t[1:], t]),
                "pn": _pick(rng, [f"{'y' if yp else ''}p{n}", n[1:], n]),
                "pn_ost_uzliny_poz": min(_positive_nodes(rng, n), examined),
                "pn_ost_uzliny_vys": examined,
                "pm": f"p{m}" if m != "M0" and rng.random() < 0.3 else None,
            }
        for k in range(int(rng.integers(1, 4))):
            rows.append({**patient, **tumour, "id_vzorku": f"{case_id}-{k + 1:02d}"})
    # a few values the parsers do not know
    if rng.random() < 0.01:
        rows[0]["kras"] = "viz zpráva"
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path, help="Output .xlsx path")
    parser.add_argument("--patients", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = [row for i in range(args.patients) for row in simulate_patient(rng, i)]
    df = pd.DataFrame(rows).reindex(columns=HEADER)
    year = pd.to_datetime(df["datum_diagnozy"], format="mixed", dayfirst=True).dt.year

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out) as writer:
        # Title line above the header, as in real exports
        df[year < 2018].to_excel(writer, sheet_name="2012-2017", startrow=2, index=False)
        writer.sheets["2012-2017"].cell(row=1, column=1, value="Export z NOR - kolorektum")
        df[year >= 2018].to_excel(writer, sheet_name="2018-2022", index=False)
        pd.DataFrame({"pole": ["pt", "pn"], "vyznam": ["primární nádor", "uzliny"]}).to_excel(
            writer, sheet_name="Legenda", index=False
        )
    print(f"Wrote {len(df)} rows of {args.patients} patients -> {args.out}")


if __name__ == "__main__":
    main()

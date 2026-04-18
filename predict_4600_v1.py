#!/usr/bin/env python3

"""
Predlog narednog izvlačenja iz jednog CSV 
sa 4600 izvucenih kombinacija.
"""


from __future__ import annotations

import csv
import sys
from collections import Counter
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

N_NUM = 39
K_PICK = 7

# Koliko najbližih istorijskih „momentum“ parova koristiti u pravilu 3 (leks-rang analogija).
RANK_STEP_ANALOGY_K: int = 9

DEFAULT_4600_CSV = Path("/data/loto7hh_4600_k31.csv")

# ---------------------------------------------------------------------------
# Leksikografski indeks 0 .. C(39,7)-1
# ---------------------------------------------------------------------------

def lex_rank(combo: Tuple[int, ...], n: int = N_NUM, k: int = K_PICK) -> int:
    r = 0
    prev = 0
    for i in range(k):
        x = combo[i]
        for val in range(prev + 1, x):
            r += comb(n - val, k - i - 1)
        prev = x
    return r


def unrank_lex(r: int, n: int = N_NUM, k: int = K_PICK) -> Tuple[int, ...]:
    rr = int(r)
    out: List[int] = []
    x = 1
    for i in range(k):
        while x <= n:
            c = comb(n - x, k - i - 1)
            if c <= rr:
                rr -= c
                x += 1
            else:
                break
        out.append(x)
        x += 1
    return tuple(out)


def load_rows(csv_path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
        for row in r:
            if not row:
                continue
            if row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(7)])
    return np.array(rows, dtype=int)


def validate_combo(t: Tuple[int, ...]) -> None:
    if len(t) != K_PICK:
        raise ValueError("mora biti 7 brojeva")
    if sorted(t) != list(t) or len(set(t)) != K_PICK:
        raise ValueError("mora biti strogo rastući, bez ponavljanja")
    if min(t) < 1 or max(t) > N_NUM:
        raise ValueError("van opsega 1..39")


# ---------------------------------------------------------------------------
# Markov reda 2 (pomoćna funkcija — npr. ako ikad proslediš duži trening van main-a)
# ---------------------------------------------------------------------------

Row = Tuple[int, ...]

_FULL_ROW_MEMO: Dict[Tuple[int, bytes], Row] = {}
_FULL_ROW_RULE: Dict[Tuple[int, bytes], str] = {}


def build_order2_map(train: np.ndarray) -> Dict[Tuple[Row, Row], Row]:
    out: Dict[Tuple[Row, Row], Row] = {}
    for i in range(train.shape[0] - 2):
        a = tuple(int(x) for x in train[i])
        b = tuple(int(x) for x in train[i + 1])
        nxt = tuple(int(x) for x in train[i + 2])
        key = (a, b)
        if key in out and out[key] != nxt:
            raise ValueError(f"Markov2 nije jednoznačan za ključ (red {i})")
        out[key] = nxt
    return out


def predict_next_row_order2(history: np.ndarray, train: np.ndarray) -> Row:
    if history.shape[0] < 2:
        raise ValueError("istorija mora imati bar 2 reda")
    if train.shape[0] < 3:
        raise ValueError("trening mora imati bar 3 reda za Markov2")
    m = build_order2_map(train)
    key = (
        tuple(int(x) for x in history[-2]),
        tuple(int(x) for x in history[-1]),
    )
    if key not in m:
        raise KeyError(
            "Par (predposlednji, poslednji) nije u tabeli — trening mora da sadrži "
            "nastavak niza (bar jedan red više od istorije)."
        )
    out = m[key]
    validate_combo(out)
    return out


def predict_next_row_markov(history: np.ndarray, train: np.ndarray) -> Tuple[int, ...]:
    return predict_next_row_order2(history, train)


# ---------------------------------------------------------------------------
# Predikcija iz istorije: lanac pravila (uvek jedan predlog)
# ---------------------------------------------------------------------------

def _predict_markov1_mode_after_last_row(history: np.ndarray) -> Optional[Row]:
    """Svi i gde je red[i+1] == poslednji red; predlog iz red[i+2]. Moda + tie-break po leks-rangu."""
    n = history.shape[0]
    if n < 3:
        return None
    b = history[-1]
    succ: List[Row] = []
    for i in range(n - 2):
        if np.array_equal(history[i + 1], b):
            succ.append(tuple(int(x) for x in history[i + 2]))
    if not succ:
        return None
    cnt = Counter(succ)
    best = max(cnt.values())
    cand = [t for t, v in cnt.items() if v == best]
    if len(cand) == 1:
        return cand[0]
    r_last = lex_rank(tuple(int(x) for x in history[-1]))
    return min(cand, key=lambda t: abs(lex_rank(t) - r_last))


def _predict_rank_step_analogy(history: np.ndarray, k_neighbors: int = RANK_STEP_ANALOGY_K) -> Row:
    """Par uzastopnih koraka u leks-rangu kao 'stanje'; medijana sledećeg koraka kod K najbližih stanja u istoriji."""
    mx = comb(N_NUM, K_PICK) - 1
    last_t = tuple(int(x) for x in history[-1])
    n = history.shape[0]
    ranks = np.array(
        [lex_rank(tuple(int(x) for x in history[i])) for i in range(n)],
        dtype=np.float64,
    )
    d1q = ranks[-1] - ranks[-2]
    d2q = ranks[-2] - ranks[-3]
    s1 = abs(d1q) + 1.0
    s2 = abs(d2q) + 1.0
    scored: List[Tuple[float, float]] = []
    # i+1 <= n-2 ⇒ ne koristimo prelaz koji završava na poslednjem redu (n−1).
    for i in range(2, n - 2):
        e1 = ranks[i] - ranks[i - 1]
        e2 = ranks[i - 1] - ranks[i - 2]
        dist = ((e1 - d1q) / s1) ** 2 + ((e2 - d2q) / s2) ** 2
        next_delta = ranks[i + 1] - ranks[i]
        scored.append((dist, next_delta))
    scored.sort(key=lambda x: x[0])
    take = min(k_neighbors, len(scored))
    deltas = [scored[j][1] for j in range(take)]
    pred_d = float(np.median(deltas))
    r_last = int(ranks[-1])
    r_next = int(round(r_last + pred_d))
    r_next = max(0, min(mx, r_next))
    out = unrank_lex(r_next)
    if out == last_t:
        return _predict_lex_rank_median_delta(history)
    return out


def _predict_lex_rank_median_delta(history: np.ndarray) -> Row:
    """r_{t+1} ≈ unrank( rank(r_t) + medijana( rank_i - rank_{i-1} ) ); nikad ne vraća istu kombinaciju kao poslednji poznati red."""
    mx = comb(N_NUM, K_PICK) - 1
    last_t = tuple(int(x) for x in history[-1])
    ranks = np.array(
        [lex_rank(tuple(int(x) for x in history[i])) for i in range(history.shape[0])],
        dtype=np.int64,
    )
    deltas = np.diff(ranks)
    r_last = int(ranks[-1])
    d = int(round(float(np.median(deltas))))
    r_next = max(0, min(mx, r_last + d))
    out = unrank_lex(r_next)
    if out != last_t:
        return out
    # Medijana koraka ≈ 0 ili slučajno isti leks-rang — pomak dok ne dobijemo drugu kombinaciju od zadnje poznate.
    for eps in (1, -1, 2, -2, 3, -3, 10, -10):
        r2 = max(0, min(mx, r_last + eps))
        cand = unrank_lex(r2)
        if cand != last_t:
            return cand
    return unrank_lex(min(mx, r_last + 1))


def _full_row_for_history(history: np.ndarray) -> Row:
    key_mem = (history.shape[0], history.tobytes())
    if key_mem in _FULL_ROW_MEMO:
        return _FULL_ROW_MEMO[key_mem]

    n = history.shape[0]
    if n < 3:
        raise ValueError("potrebno bar 3 reda")

    a = tuple(int(x) for x in history[-2])
    b = tuple(int(x) for x in history[-1])
    succ_pair: List[Row] = []
    for i in range(n - 2):
        if tuple(int(x) for x in history[i]) == a and tuple(int(x) for x in history[i + 1]) == b:
            if i + 2 < n:
                succ_pair.append(tuple(int(x) for x in history[i + 2]))

    if succ_pair:
        uniq = list(dict.fromkeys(succ_pair))
        if len(uniq) != 1:
            raise ValueError(f"Za isti par (predposlednji,poslednji) različiti nastavci: {uniq}")
        out = uniq[0]
        rule = "pravilo 1: ponovljen par redova (isti nastavak u istoriji)"
    else:
        m1 = _predict_markov1_mode_after_last_row(history)
        if m1 is not None:
            out = m1
            rule = "pravilo 2: moda nastavka posle istog poslednjeg reda (Markov1 iz istorije)"
        elif n >= 5:
            out = _predict_rank_step_analogy(history)
            rule = (
                "pravilo 3: analogija uzastopnih koraka leks-ranga "
                f"(K={RANK_STEP_ANALOGY_K} najbližih situacija u istoriji)"
            )
        else:
            out = _predict_lex_rank_median_delta(history)
            rule = "pravilo 4: leks-rang + medijana koraka na celoj istoriji (kratka istorija)"

    last_t = tuple(int(x) for x in history[-1])
    if out == last_t:
        # Cilj je naredno izvlačenje, ne ponavljanje zadnjeg poznatog reda iz CSV-a.
        out = _predict_lex_rank_median_delta(history)
        rule = "pravilo 4: leks-rang + medijana koraka (predlog bi inače bio isti kao zadnji poznat red)"

    validate_combo(out)
    _FULL_ROW_MEMO[key_mem] = out
    _FULL_ROW_RULE[key_mem] = rule
    return out


def formula_num1(vert1: np.ndarray, history: np.ndarray) -> int:
    return _full_row_for_history(history)[0]


def formula_num2(vert1: np.ndarray, vert2: np.ndarray, num1: int, history: np.ndarray) -> int:
    return _full_row_for_history(history)[1]


def formula_num3(
    vert1: np.ndarray, vert2: np.ndarray, vert3: np.ndarray,
    num1: int, num2: int, history: np.ndarray,
) -> int:
    return _full_row_for_history(history)[2]


def formula_num4(
    vert1: np.ndarray, vert2: np.ndarray, vert3: np.ndarray, vert4: np.ndarray,
    num1: int, num2: int, num3: int, history: np.ndarray,
) -> int:
    return _full_row_for_history(history)[3]


def formula_num5(
    verts: List[np.ndarray],
    prefix: Tuple[int, int, int, int],
    history: np.ndarray,
) -> int:
    return _full_row_for_history(history)[4]


def formula_num6(
    verts: List[np.ndarray],
    prefix: Tuple[int, int, int, int, int],
    history: np.ndarray,
) -> int:
    return _full_row_for_history(history)[5]


def formula_num7(
    verts: List[np.ndarray],
    prefix: Tuple[int, int, int, int, int, int],
    history: np.ndarray,
) -> int:
    return _full_row_for_history(history)[6]


def predict_next_row_from_formulas(history: np.ndarray) -> Tuple[int, ...]:
    out = _full_row_for_history(history)
    validate_combo(out)
    return out


def predict_row_from_lex_rank(rank_next: int) -> Tuple[int, ...]:
    mx = comb(N_NUM, K_PICK) - 1
    if rank_next < 0 or rank_next > mx:
        raise ValueError(f"rank van [0,{mx}]")
    return unrank_lex(rank_next)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
print()
def main(argv: List[str]) -> int:
    if len(argv) >= 2 and argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0

    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_4600_CSV
    history = load_rows(path)
    if history.shape[0] != 4600:
        print("Očekuje se tačno 4600 redova, ima:", history.shape[0])
        return 1

    print("CSV:", path)
    print("Učitano poznatih redova (istorija):", history.shape[0])
    last_known = tuple(int(x) for x in history[-1])
    print("Poslednji poznat red (zadnji u CSV fajlu):", last_known)

    try:
        pred = predict_next_row_from_formulas(history)
    except ValueError as e:
        print(e)
        return 2

    key_mem = (history.shape[0], history.tobytes())
    print("pravilo:", _FULL_ROW_RULE.get(key_mem, "?"))
    print("Predlog sledećeg izvlačenja (naredni red u nizu, posle zadnjeg iz CSV-a):", pred)
    if pred == last_known:
        print("UPOZORENJE: predlog je isti kao zadnji poznat red — proveri podatke ili pravila.")
    print()
    return 0
print()

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


"""
CSV: /data/loto7hh_4600_k31.csv
Učitano poznatih redova (istorija): 4600
Poslednji poznat red (zadnji u CSV fajlu): (1, 4, 11, 14, 15, 19, 25)
pravilo: pravilo 2: vertikalni lanac (Num_k | Num_1…Num_{k-1}, bez Markov/kNN)

Predlog sledećeg izvlačenja 
(naredni red u nizu, posle zadnjeg iz CSV): 
(2, x, 11, y, 28, z, 37)
"""

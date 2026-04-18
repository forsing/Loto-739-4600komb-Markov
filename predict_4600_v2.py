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

# Broj najbližih parova redova u pravilu 3 (kNN → težinski glas nad sledećim redom).
KNN_PAIR_K: int = 20

DEFAULT_4600_CSV = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4600_k31.csv")

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


def _slot_bounds(prev: int, slot_index: int) -> Tuple[int, int]:
    """slot_index 0..6 za Num_1..Num_7; prev je prethodni broj ili 0 pre prvog."""
    lo = prev + 1 if prev > 0 else 1
    hi = 33 + slot_index
    return lo, hi


def _conditional_column_sample(history: np.ndarray, slot_k: int, prefix: Tuple[int, ...]) -> np.ndarray:
    """Vrednosti Num_{k+1} iz redova koji se poklapaju na prvih len(prefix) pozicija; backoff kraćenjem prefiksa."""
    plen = len(prefix)
    for back in range(plen + 1):
        pfx = prefix if back == 0 else prefix[: plen - back]
        m = np.ones(history.shape[0], dtype=bool)
        for t, val in enumerate(pfx):
            m &= history[:, t] == val
        col = history[m, slot_k]
        if col.size > 0:
            return col
    return history[:, slot_k]


def _pick_mode_in_range(values: np.ndarray, lo: int, hi: int, forbid: Optional[int] = None) -> int:
    c = Counter(int(x) for x in values if lo <= int(x) <= hi and (forbid is None or int(x) != forbid))
    if not c:
        c = Counter(int(x) for x in values if lo <= int(x) <= hi)
    if not c:
        return min(hi, max(lo, (lo + hi) // 2))
    best = max(c.values())
    cand = sorted(v for v, ct in c.items() if ct == best)
    return cand[len(cand) // 2]


def _predict_vertical_conditional_chain(history: np.ndarray, forbid_at_slot: Optional[int] = None) -> Row:
    """
    Vertikalno: Num_1, zatim Num_2|Num_1, … iz empirijskih uslovnih raspodela u istoriji.
    Opseg po pozicijama poštuje sortiranu 7-kombinaciju iz {1..39}.
    """
    built: List[int] = []
    for k in range(K_PICK):
        prev = built[-1] if built else 0
        lo, hi = _slot_bounds(prev, k)
        col = _conditional_column_sample(history, k, tuple(built))
        fv = None
        if forbid_at_slot is not None and k == forbid_at_slot:
            fv = int(history[-1, k])
        v = _pick_mode_in_range(col, lo, hi, forbid=fv)
        built.append(v)
    return tuple(built)


def _predict_knn_pair_weighted_successor(history: np.ndarray) -> Row:
    """
    Horizontalni kontekst: poslednja dva reda. Nadji K najbližih istorijskih parova
    (red_i, red_{i+1}) po L1 na 14D; glasaj za red_{i+2} sa težinom 1/(d+1).
    """
    n = history.shape[0]
    last_t = tuple(int(x) for x in history[-1])
    if n < 3:
        return _predict_vertical_conditional_not_equal_last(history)
    q = np.concatenate([history[-2].astype(np.float64), history[-1].astype(np.float64)])
    scored: List[Tuple[float, int]] = []
    for i in range(n - 2):
        v = np.concatenate([history[i].astype(np.float64), history[i + 1].astype(np.float64)])
        d = float(np.sum(np.abs(v - q)))
        scored.append((d, i))
    scored.sort(key=lambda x: x[0])
    take = min(KNN_PAIR_K, len(scored))
    ctr: Counter[Row] = Counter()
    for j in range(take):
        d, i = scored[j]
        tup = tuple(int(x) for x in history[i + 2])
        ctr[tup] += 1.0 / (d + 1.0)
    ordered = sorted(ctr.items(), key=lambda x: (-x[1], x[0]))
    for tup, _ in ordered:
        if tup != last_t:
            return tup
    return ordered[0][0]


def _predict_vertical_conditional_not_equal_last(history: np.ndarray) -> Row:
    """Isto kao lanac; ako bi ispalo kao poslednji poznati red, zabrani vrednost na prvoj poziciji gde ima alternative."""
    last_t = tuple(int(x) for x in history[-1])
    base = _predict_vertical_conditional_chain(history)
    if base != last_t:
        return base
    for ks in range(K_PICK):
        alt = _predict_vertical_conditional_chain(history, forbid_at_slot=ks)
        if alt != last_t:
            return alt
    return base


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
        else:
            out = _predict_knn_pair_weighted_successor(history)
            rule = (
                f"pravilo 3: kNN par redova (K={KNN_PAIR_K}, L1) — težinski glas nad sledećim redom"
            )

    last_t = tuple(int(x) for x in history[-1])
    if out == last_t:
        out = _predict_vertical_conditional_not_equal_last(history)
        rule = (
            "pravilo 4: vertikalni lanac — izbegnut predlog identičan poslednjem poznatom redu"
        )

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
(10, 11, 13, 20, 27, 28, 38)
"""

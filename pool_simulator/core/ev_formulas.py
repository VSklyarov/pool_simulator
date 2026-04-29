"""EV calculation formulas for different analytical approaches.

Each formula computes the total EV (in dollars) for a batch of simulated
tournaments. The *Result* is always the same::

    Result_i = buyin * part_m * [((ChipNW + S)/(3*S)) * w1_m
               + ((2*S - ChipNW)/(3*S)) * w23_m  -  1]

Swing = sum(Result) - EV, where EV depends on the chosen formula.

Additive formulas (AvgEV, MultEV, MultEV50, MultEV4) compute EV per-tournament
and sum it.  Non-additive formulas (AW_EV, AvgEV4) depend on aggregate
statistics of the entire batch.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass

import numpy as np

from pool_simulator.core.models import Multiplier, Setup


class EVFormula(Enum):
    AVG_EV = "AvgEV"
    MULT_EV = "MultEV"
    MULT_EV_50 = "MultEV50"
    MULT_EV_4 = "MultEV4"
    AW_EV = "AW EV"
    AVG_EV_4 = "AvgEV4"


def is_additive(formula: EVFormula) -> bool:
    return formula in (EVFormula.AVG_EV, EVFormula.MULT_EV,
                       EVFormula.MULT_EV_50, EVFormula.MULT_EV_4)


JACKPOT_THRESHOLD = 1000.0


@dataclass
class StructureParams:
    """Pre-computed per-structure constants used across EV formulas."""

    buyin: float
    n_mult: int
    mult_values: np.ndarray    # (K,)
    w1: np.ndarray             # (K,)
    w23: np.ndarray            # (K,)
    probs: np.ndarray          # (K,)
    parts: np.ndarray          # (K,)
    stacks: np.ndarray         # (K,) int
    min_stack: int
    has_variable_stacks: bool

    # Aggregated metrics (computed lazily)
    w1_avg: float = 0.0
    w23_avg: float = 0.0
    avg_stack: float = 0.0

    # x4+ aggregates
    w1_avg_4plus: float = 0.0
    w23_avg_4plus: float = 0.0
    avg_stack_4plus: float = 0.0

    # x50-x100 aggregates
    w1_avg_50_100: float = 0.0
    w23_avg_50_100: float = 0.0
    avg_stack_50_100: float = 0.0

    # Jackpot info
    jackpot_idx: int = -1
    has_jackpot: bool = False


def build_structure_params(
    setup: Setup,
    participations: list[float],
) -> StructureParams:
    """Pre-compute all structural constants needed by EV formulas."""
    mults = setup.multipliers
    n = len(mults)
    buyin = setup.buyin

    mult_values = np.array([m.multiplier_value for m in mults])
    w1 = np.array([m.w1 for m in mults])
    w23 = np.array([m.w23 for m in mults])
    probs = np.array([m.probability for m in mults])
    parts = np.array(participations, dtype=np.float64)
    stacks = np.array([m.stack_size for m in mults], dtype=np.int64)
    min_stack = int(stacks.min()) if n > 0 else 0

    sp = StructureParams(
        buyin=buyin, n_mult=n,
        mult_values=mult_values, w1=w1, w23=w23,
        probs=probs, parts=parts, stacks=stacks,
        min_stack=min_stack,
        has_variable_stacks=setup.has_variable_stacks,
    )

    # --- AvgEV aggregates: sumproduct(metric * prob * part) ---
    wp = probs * parts  # not normalized — direct sumproduct as described
    sp.w1_avg = float(np.dot(w1, wp))
    sp.w23_avg = float(np.dot(w23, wp))
    sp.avg_stack = float(np.dot(stacks.astype(np.float64), probs))

    # --- x4+ aggregates ---
    mask_4plus = mult_values >= 4
    if mask_4plus.any():
        p4 = probs[mask_4plus]
        sum_p4 = p4.sum()
        if sum_p4 > 0:
            sp.w1_avg_4plus = float(np.dot(w1[mask_4plus] * parts[mask_4plus], p4) / sum_p4)
            sp.w23_avg_4plus = float(np.dot(w23[mask_4plus] * parts[mask_4plus], p4) / sum_p4)
            sp.avg_stack_4plus = float(np.dot(stacks[mask_4plus].astype(np.float64), p4) / sum_p4)

    # --- x50-x100 aggregates ---
    mask_50_100 = (mult_values >= 50) & (mult_values <= 100)
    if mask_50_100.any():
        p50 = probs[mask_50_100]
        sum_p50 = p50.sum()
        if sum_p50 > 0:
            sp.w1_avg_50_100 = float(np.dot(w1[mask_50_100] * parts[mask_50_100], p50) / sum_p50)
            sp.w23_avg_50_100 = float(np.dot(w23[mask_50_100] * parts[mask_50_100], p50) / sum_p50)
            sp.avg_stack_50_100 = float(np.dot(stacks[mask_50_100].astype(np.float64), p50) / sum_p50)

    # --- Jackpot ---
    max_mult = float(mult_values.max()) if n > 0 else 0
    if max_mult > JACKPOT_THRESHOLD:
        sp.jackpot_idx = int(np.argmax(mult_values))
        sp.has_jackpot = parts[sp.jackpot_idx] > 0

    return sp


# ---------------------------------------------------------------------------
# Per-tournament EV helpers
# ---------------------------------------------------------------------------

def _ev_avg_single(chip_ev: float, sp: StructureParams) -> float:
    """AvgEV for a single tournament."""
    S = sp.avg_stack
    if S == 0:
        return 0.0
    return sp.buyin * (
        (chip_ev + S) / (3.0 * S) * sp.w1_avg
        + (2.0 * S - chip_ev) / (3.0 * S) * sp.w23_avg
        - 1.0
    )


def _ev_mult_single(chip_ev: float, mult_idx: int, sp: StructureParams) -> float:
    """MultEV for a single tournament whose multiplier is at mult_idx.

    If the multiplier is a jackpot (>x1000) the EV in that tournament
    is computed via AvgEV instead.
    """
    mv = sp.mult_values[mult_idx]

    if mv > JACKPOT_THRESHOLD:
        return _ev_avg_single(chip_ev, sp)

    S = sp.stacks[mult_idx]
    part = sp.parts[mult_idx]
    w1_m = sp.w1[mult_idx]
    w23_m = sp.w23[mult_idx]

    base = sp.buyin * part * (
        (chip_ev + S) / (3.0 * S) * w1_m
        + (2.0 * S - chip_ev) / (3.0 * S) * w23_m
        - 1.0
    )

    jackpot_ev = _jackpot_addon(chip_ev, sp)
    return base + jackpot_ev


def _jackpot_addon(chip_ev: float, sp: StructureParams) -> float:
    """JackpotEV addon used in MultEV and MultEV50."""
    if not sp.has_jackpot:
        return 0.0
    ji = sp.jackpot_idx
    S = sp.avg_stack
    if S == 0:
        return 0.0
    return (
        sp.probs[ji] * sp.parts[ji] * sp.buyin
        * (
            (chip_ev + S) / (3.0 * S) * sp.w1[ji]
            + (2.0 * S - chip_ev) / (3.0 * S) * sp.w23[ji]
        )
    )


def _ev_multev50_single(chip_ev: float, mult_idx: int, sp: StructureParams) -> float:
    """MultEV50 for a single tournament."""
    mv = sp.mult_values[mult_idx]

    if mv > JACKPOT_THRESHOLD:
        return _ev_avg_single(chip_ev, sp)

    if 50 <= mv <= 100:
        S = sp.avg_stack_50_100 if sp.avg_stack_50_100 > 0 else sp.stacks[mult_idx]
        w1_m = sp.w1_avg_50_100
        w23_m = sp.w23_avg_50_100
        part = 1.0
    else:
        S = sp.stacks[mult_idx]
        w1_m = sp.w1[mult_idx]
        w23_m = sp.w23[mult_idx]
        part = sp.parts[mult_idx]

    base = sp.buyin * part * (
        (chip_ev + S) / (3.0 * S) * w1_m
        + (2.0 * S - chip_ev) / (3.0 * S) * w23_m
        - 1.0
    )

    jackpot_ev = _jackpot_addon(chip_ev, sp)
    return base + jackpot_ev


def _ev_multev4_single(chip_ev: float, mult_idx: int, sp: StructureParams) -> float:
    """MultEV4 for a single tournament."""
    mv = sp.mult_values[mult_idx]

    if mv >= 4:
        S = sp.avg_stack_4plus if sp.avg_stack_4plus > 0 else sp.stacks[mult_idx]
        return sp.buyin * (
            (chip_ev + S) / (3.0 * S) * sp.w1_avg_4plus
            + (2.0 * S - chip_ev) / (3.0 * S) * sp.w23_avg_4plus
            - 1.0
        )

    # x2, x3: use MultEV logic (no separate jackpot addon since it's in avg)
    S = sp.stacks[mult_idx]
    part = sp.parts[mult_idx]
    return sp.buyin * part * (
        (chip_ev + S) / (3.0 * S) * sp.w1[mult_idx]
        + (2.0 * S - chip_ev) / (3.0 * S) * sp.w23[mult_idx]
        - 1.0
    )


# ---------------------------------------------------------------------------
# Vectorized additive EV computation
# ---------------------------------------------------------------------------

def compute_ev_additive_batch(
    mult_indices: np.ndarray,
    chip_evs: np.ndarray,
    sp: StructureParams,
    formula: EVFormula,
) -> np.ndarray:
    """Compute per-tournament EV for additive formulas (vectorized).

    Returns an array of per-tournament EV values (same length as mult_indices).
    """
    n = len(mult_indices)
    ev_out = np.empty(n, dtype=np.float64)

    if formula == EVFormula.AVG_EV:
        S = sp.avg_stack
        if S == 0:
            return np.zeros(n)
        ev_out[:] = sp.buyin * (
            (chip_evs + S) / (3.0 * S) * sp.w1_avg
            + (2.0 * S - chip_evs) / (3.0 * S) * sp.w23_avg
            - 1.0
        )
        return ev_out

    mi_stacks = sp.stacks[mult_indices].astype(np.float64)
    mi_w1 = sp.w1[mult_indices]
    mi_w23 = sp.w23[mult_indices]
    mi_parts = sp.parts[mult_indices]
    mi_mvals = sp.mult_values[mult_indices]

    if formula == EVFormula.MULT_EV:
        is_jackpot = mi_mvals > JACKPOT_THRESHOLD
        S = np.where(is_jackpot, sp.avg_stack, mi_stacks)
        S = np.where(S == 0, 1.0, S)

        w1_use = np.where(is_jackpot, sp.w1_avg, mi_w1)
        w23_use = np.where(is_jackpot, sp.w23_avg, mi_w23)
        part_use = np.where(is_jackpot, 1.0, mi_parts)

        ev_out[:] = sp.buyin * part_use * (
            (chip_evs + S) / (3.0 * S) * w1_use
            + (2.0 * S - chip_evs) / (3.0 * S) * w23_use
            - 1.0
        )

        if sp.has_jackpot:
            ji = sp.jackpot_idx
            S_avg = sp.avg_stack if sp.avg_stack > 0 else 1.0
            addon = (
                sp.probs[ji] * sp.parts[ji] * sp.buyin
                * (
                    (chip_evs + S_avg) / (3.0 * S_avg) * sp.w1[ji]
                    + (2.0 * S_avg - chip_evs) / (3.0 * S_avg) * sp.w23[ji]
                )
            )
            ev_out[~is_jackpot] += addon[~is_jackpot]

        return ev_out

    if formula == EVFormula.MULT_EV_50:
        is_jackpot = mi_mvals > JACKPOT_THRESHOLD
        is_50_100 = (mi_mvals >= 50) & (mi_mvals <= 100) & ~is_jackpot
        is_normal = ~is_jackpot & ~is_50_100

        S = np.where(is_jackpot, sp.avg_stack,
                     np.where(is_50_100,
                              sp.avg_stack_50_100 if sp.avg_stack_50_100 > 0 else mi_stacks,
                              mi_stacks))
        S = np.where(S == 0, 1.0, S)

        w1_use = np.where(is_jackpot, sp.w1_avg,
                          np.where(is_50_100, sp.w1_avg_50_100, mi_w1))
        w23_use = np.where(is_jackpot, sp.w23_avg,
                           np.where(is_50_100, sp.w23_avg_50_100, mi_w23))
        part_use = np.where(is_jackpot, 1.0,
                            np.where(is_50_100, 1.0, mi_parts))

        ev_out[:] = sp.buyin * part_use * (
            (chip_evs + S) / (3.0 * S) * w1_use
            + (2.0 * S - chip_evs) / (3.0 * S) * w23_use
            - 1.0
        )

        if sp.has_jackpot:
            ji = sp.jackpot_idx
            S_avg = sp.avg_stack if sp.avg_stack > 0 else 1.0
            addon = (
                sp.probs[ji] * sp.parts[ji] * sp.buyin
                * (
                    (chip_evs + S_avg) / (3.0 * S_avg) * sp.w1[ji]
                    + (2.0 * S_avg - chip_evs) / (3.0 * S_avg) * sp.w23[ji]
                )
            )
            ev_out[~is_jackpot] += addon[~is_jackpot]

        return ev_out

    if formula == EVFormula.MULT_EV_4:
        is_4plus = mi_mvals >= 4

        S_4p = sp.avg_stack_4plus if sp.avg_stack_4plus > 0 else 1.0
        S = np.where(is_4plus, S_4p, mi_stacks)
        S = np.where(S == 0, 1.0, S)

        w1_use = np.where(is_4plus, sp.w1_avg_4plus, mi_w1)
        w23_use = np.where(is_4plus, sp.w23_avg_4plus, mi_w23)
        part_use = np.where(is_4plus, 1.0, mi_parts)

        ev_out[:] = sp.buyin * part_use * (
            (chip_evs + S) / (3.0 * S) * w1_use
            + (2.0 * S - chip_evs) / (3.0 * S) * w23_use
            - 1.0
        )
        return ev_out

    return np.zeros(n)


# ---------------------------------------------------------------------------
# Non-additive EV: AW EV
# ---------------------------------------------------------------------------

def compute_ev_aw(
    mult_indices: np.ndarray,
    chip_evs: np.ndarray,
    sp: StructureParams,
) -> float:
    """Compute total EV for the batch using AW EV formula.

    Returns a scalar (total EV for the entire batch in dollars).
    """
    n = len(mult_indices)
    if n == 0:
        return 0.0

    mi_stacks = sp.stacks[mult_indices].astype(np.float64)
    mi_mvals = sp.mult_values[mult_indices]

    # Normalize ChipEV to min_stack if variable stacks
    if sp.has_variable_stacks and sp.min_stack > 0:
        chip_evs_norm = chip_evs * sp.min_stack / mi_stacks
    else:
        chip_evs_norm = chip_evs.copy()

    # Weights: x2 -> 2, x3 -> 3, x4+ -> sumproduct(total_adj * prob_adj) / sum(prob_adj)
    total_adj = sp.parts * (sp.w1 - sp.w23)
    prob_adj = sp.probs * sp.parts
    mask_4plus = sp.mult_values >= 4
    if mask_4plus.any() and prob_adj[mask_4plus].sum() > 0:
        w4plus = float(np.dot(total_adj[mask_4plus], prob_adj[mask_4plus])
                       / prob_adj[mask_4plus].sum())
    else:
        w4plus = 0.0

    weights_per_mult = np.zeros(sp.n_mult, dtype=np.float64)
    for k in range(sp.n_mult):
        mv = sp.mult_values[k]
        if mv < 4:
            weights_per_mult[k] = mv
        else:
            weights_per_mult[k] = w4plus

    # Group by multiplier: mean normalized ChipEV and count
    mi_weights = weights_per_mult[mult_indices]
    unique_k = np.unique(mult_indices)
    weighted_num = 0.0
    weighted_den = 0.0
    for k in unique_k:
        mask_k = mult_indices == k
        count_k = mask_k.sum()
        mean_cev_k = float(chip_evs_norm[mask_k].mean())
        w_k = weights_per_mult[k]
        weighted_num += mean_cev_k * w_k * count_k
        weighted_den += w_k * count_k

    if weighted_den == 0:
        return 0.0
    wcev = weighted_num / weighted_den

    # Use min_stack (not avg_stack) as S for the formula
    S = float(sp.min_stack) if sp.min_stack > 0 else sp.avg_stack
    if S == 0:
        return 0.0

    ev_total = sp.buyin * n * (
        (wcev + S) / (3.0 * S) * sp.w1_avg
        + (2.0 * S - wcev) / (3.0 * S) * sp.w23_avg
        - 1.0
    )
    return float(ev_total)


# ---------------------------------------------------------------------------
# Non-additive EV: AvgEV4
# ---------------------------------------------------------------------------

def compute_ev_avgev4(
    mult_indices: np.ndarray,
    chip_evs: np.ndarray,
    sp: StructureParams,
) -> float:
    """Compute total EV for the batch using AvgEV4 formula.

    Returns a scalar (total EV for the entire batch in dollars).
    """
    n = len(mult_indices)
    if n == 0:
        return 0.0

    total_ev = 0.0

    for k in range(sp.n_mult):
        mv = sp.mult_values[k]
        # Theoretical count
        theo_count = n * sp.probs[k]

        mask_k = mult_indices == k
        if mask_k.any():
            mean_cev_k = float(chip_evs[mask_k].mean())
        else:
            mean_cev_k = 0.0

        if mv < 4:
            # x2, x3: MultEV formula * theoretical count
            S = float(sp.stacks[k])
            if S == 0:
                continue
            part = sp.parts[k]
            ev_k = sp.buyin * part * (
                (mean_cev_k + S) / (3.0 * S) * sp.w1[k]
                + (2.0 * S - mean_cev_k) / (3.0 * S) * sp.w23[k]
                - 1.0
            ) * theo_count
        else:
            # x4+: AvgEV formula with x4+ aggregates * theoretical count
            S = sp.avg_stack_4plus if sp.avg_stack_4plus > 0 else float(sp.stacks[k])
            if S == 0:
                continue
            ev_k = sp.buyin * (
                (mean_cev_k + S) / (3.0 * S) * sp.w1_avg_4plus
                + (2.0 * S - mean_cev_k) / (3.0 * S) * sp.w23_avg_4plus
                - 1.0
            ) * theo_count

        total_ev += ev_k

    return float(total_ev)


# ---------------------------------------------------------------------------
# Result computation (same for all formulas)
# ---------------------------------------------------------------------------

def compute_result_batch(
    mult_indices: np.ndarray,
    chip_nws: np.ndarray,
    sp: StructureParams,
) -> np.ndarray:
    """Compute per-tournament Result (vectorized).

    Result_i = buyin * part_m * [((ChipNW + S)/(3S)) * w1 + ((2S - ChipNW)/(3S)) * w23 - 1]
    """
    mi_stacks = sp.stacks[mult_indices].astype(np.float64)
    mi_stacks = np.where(mi_stacks == 0, 1.0, mi_stacks)
    mi_w1 = sp.w1[mult_indices]
    mi_w23 = sp.w23[mult_indices]
    mi_parts = sp.parts[mult_indices]

    return sp.buyin * mi_parts * (
        (chip_nws + mi_stacks) / (3.0 * mi_stacks) * mi_w1
        + (2.0 * mi_stacks - chip_nws) / (3.0 * mi_stacks) * mi_w23
        - 1.0
    )


# ---------------------------------------------------------------------------
# Unified EV dispatcher
# ---------------------------------------------------------------------------

def compute_total_ev(
    mult_indices: np.ndarray,
    chip_evs: np.ndarray,
    sp: StructureParams,
    formula: EVFormula,
) -> float:
    """Compute total EV for a batch of tournaments. Returns scalar (dollars)."""
    if is_additive(formula):
        return float(compute_ev_additive_batch(mult_indices, chip_evs, sp, formula).sum())
    elif formula == EVFormula.AW_EV:
        return compute_ev_aw(mult_indices, chip_evs, sp)
    elif formula == EVFormula.AVG_EV_4:
        return compute_ev_avgev4(mult_indices, chip_evs, sp)
    return 0.0

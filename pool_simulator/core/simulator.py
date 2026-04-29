"""Monte Carlo simulation engine and analytical variance calculations.

Three simulation modes:
- **Eco**: CLT-based approximation. Replaces per-tournament sampling with
  Normal approximation of sum(ChipEV) and sum(epsilon). O(M*K) complexity.
  Fast (~seconds), sigma accuracy within ~3-5%.
- **No ChipEV Var**: Same CLT eco approximation, but ChipEV variance is
  disabled and ChipEV is fixed at the target mean from simulation params.
- **Precise**: Exact per-tournament sampling with multinomial counts and
  alias-table sampling. Multi-threaded via ThreadPoolExecutor.
  Slower but sigma accuracy within ~2%.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np

from pool_simulator.core.models import (
    DiscreteDistribution,
    Setup,
    SimulationResult,
    StackDistributions,
)
from pool_simulator.core.ev_formulas import (
    EVFormula,
    StructureParams,
    build_structure_params,
    compute_ev_additive_batch,
    compute_result_batch,
    compute_total_ev,
    is_additive,
)

CLT_THRESHOLD = 50
_MAX_ELEMENTS_PER_BATCH = 200_000_000  # ~1.6 GB of float64


def compute_analytical_sigma(
    setup: Setup,
    participations: list[float],
    mean_chip_ev: dict[int, float],
    distributions: dict[int, StackDistributions],
    ev_formula: EVFormula = EVFormula.AVG_EV,
) -> float:
    """Analytical per-tournament sigma of Swing = Result - E[Result]."""
    buyin = setup.buyin
    n_mult = len(setup.multipliers)

    probs = np.array([m.probability for m in setup.multipliers])
    ev_m = np.zeros(n_mult)
    allin_var = 0.0

    for i, m in enumerate(setup.multipliers):
        part = participations[i]
        stack = m.stack_size
        mev = mean_chip_ev.get(stack, 0.0)

        c_m = buyin * part * (m.w1 - m.w23) / (3.0 * stack)
        prob_win = (mev + stack) / (3.0 * stack)
        k_m = prob_win * m.w1 + (1.0 - prob_win) * m.w23 - 1.0
        ev_m[i] = buyin * part * k_m

        sd = distributions.get(stack)
        sigma_eps = sd.allin_stats_unconditional.get("std", 378.0) if sd else 378.0
        allin_var += probs[i] * c_m ** 2 * sigma_eps ** 2

    if ev_formula in (EVFormula.MULT_EV, EVFormula.MULT_EV_50):
        return float(np.sqrt(max(allin_var, 0.0)))

    e_total = float(np.dot(probs, ev_m))
    s_m = ev_m - e_total
    mult_var = float(np.dot(probs, s_m ** 2))

    return float(np.sqrt(max(allin_var + mult_var, 0.0)))


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _get_epsilon_std(sd: StackDistributions) -> float:
    return sd.allin_stats_unconditional.get("std", 378.0)


def _sample_allin_batch(
    chip_evs: np.ndarray,
    sd: StackDistributions,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample epsilon (ChipNW - ChipEV) for a batch of ChipEV values.

    Groups by conditional bin via argsort on int8 (radix sort, O(n)).
    """
    n = len(chip_evs)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if len(sd.conditional_bin_keys) == 0:
        return sd.allin_unconditional.sample(n, rng)

    keys = sd.conditional_bin_keys
    n_bins = len(keys)
    idx = np.searchsorted(keys, chip_evs, side="left").clip(0, n_bins - 1)
    left = (idx - 1).clip(0)
    best = np.where(
        np.abs(chip_evs - keys[left]) < np.abs(chip_evs - keys[idx]),
        left, idx,
    ).astype(np.int8)

    order = best.argsort(kind="stable")
    sorted_bins = best[order]

    eps_sorted = np.empty(n, dtype=np.float64)

    bin_ids = np.arange(n_bins, dtype=np.int8)
    starts = np.searchsorted(sorted_bins, bin_ids, side="left")
    ends = np.searchsorted(sorted_bins, bin_ids, side="right")

    for bi in range(n_bins):
        s, e = int(starts[bi]), int(ends[bi])
        cnt = e - s
        if cnt == 0:
            continue
        bin_key = int(keys[bi])
        cb = sd.conditional_bins.get(bin_key)
        if cb is not None:
            eps_sorted[s:e] = cb.distribution.sample(cnt, rng)
        else:
            eps_sorted[s:e] = sd.allin_unconditional.sample(cnt, rng)

    eps = np.empty(n, dtype=np.float64)
    eps[order] = eps_sorted
    return eps


def _segmented_sum_via_cumsum(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    cs = np.cumsum(values)
    ends = np.cumsum(counts)
    starts = ends - counts
    return np.where(
        counts > 0,
        cs[(ends - 1).clip(0)] - np.where(starts > 0, cs[(starts - 1).clip(0)], 0.0),
        0.0,
    )


def _adaptive_batch_size(n_simulations: int, n_tourneys: int, base: int) -> int:
    """Choose batch size that keeps memory usage reasonable."""
    max_by_mem = max(500, _MAX_ELEMENTS_PER_BATCH // max(n_tourneys, 1))
    return min(n_simulations, base, max_by_mem)


# ---------------------------------------------------------------------------
# Eco mode: CLT-based simulation (additive formulas)
# ---------------------------------------------------------------------------

def _precompute_distribution_moments(
    sp: StructureParams,
    distributions: dict[int, StackDistributions],
    mean_chip_ev: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute (mu_cev, sigma_cev, sigma_eps) per multiplier.

    Returns arrays of shape (K,).
    """
    K = sp.n_mult
    mu_cev = np.zeros(K, dtype=np.float64)
    sigma_cev = np.zeros(K, dtype=np.float64)
    sigma_eps = np.zeros(K, dtype=np.float64)

    for k in range(K):
        stack = int(sp.stacks[k])
        sd = distributions.get(stack)
        target_mean = mean_chip_ev.get(stack, 0.0)

        if sd is not None:
            mu_cev[k] = target_mean
            sigma_cev[k] = sd.chip_ev.std
            sigma_eps[k] = _get_epsilon_std(sd)
        else:
            mu_cev[k] = target_mean

    return mu_cev, sigma_cev, sigma_eps


def _simulate_additive_eco(
    setup: Setup,
    sp: StructureParams,
    distributions: dict[int, StackDistributions],
    mean_chip_ev: dict[int, float],
    n_tourneys: int,
    n_simulations: int,
    ev_formula: EVFormula,
    rng: np.random.Generator,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Eco-mode additive simulation using CLT approximation.

    Instead of sampling N individual ChipEV and epsilon values per
    multiplier, approximates sum(ChipEV) and sum(epsilon) as Normal
    random variables. Complexity: O(M * K).
    """
    n_mult = sp.n_mult
    buyin = setup.buyin

    probs = sp.probs.copy()
    probs /= probs.sum()

    # Precompute swing coefficients (same as precise)
    coeffs_result = np.empty(n_mult, dtype=np.float64)
    consts_result = np.empty(n_mult, dtype=np.float64)
    for i in range(n_mult):
        stack = int(sp.stacks[i])
        part = sp.parts[i]
        coeffs_result[i] = buyin * part * (sp.w1[i] - sp.w23[i]) / (3.0 * stack)
        consts_result[i] = buyin * part * (sp.w1[i] / 3.0 + 2.0 * sp.w23[i] / 3.0 - 1.0)

    ev_const = np.empty(n_mult, dtype=np.float64)
    ev_coeff = np.empty(n_mult, dtype=np.float64)
    for k in range(n_mult):
        idx_arr = np.array([k], dtype=np.intp)
        ev_at_0 = float(compute_ev_additive_batch(
            idx_arr, np.array([0.0]), sp, ev_formula)[0])
        ev_at_1 = float(compute_ev_additive_batch(
            idx_arr, np.array([1.0]), sp, ev_formula)[0])
        ev_const[k] = ev_at_0
        ev_coeff[k] = ev_at_1 - ev_at_0

    swing_const = consts_result - ev_const
    swing_cev_coeff = coeffs_result - ev_coeff
    swing_eps_coeff = coeffs_result

    mu_cev, sigma_cev, sigma_eps = _precompute_distribution_moments(
        sp, distributions, mean_chip_ev,
    )

    batch_size = min(n_simulations, 500_000)
    swings = np.empty(n_simulations, dtype=np.float64)

    for sim_start in range(0, n_simulations, batch_size):
        sim_end = min(sim_start + batch_size, n_simulations)
        B = sim_end - sim_start

        counts = rng.multinomial(n_tourneys, probs, size=B)  # (B, K)
        counts_f = counts.astype(np.float64)

        # Deterministic part: sum of per-tourney constants
        batch_swings = counts_f @ swing_const  # (B,)

        # CLT: sum_cev_k ~ Normal(n_k * mu_k, sqrt(n_k) * sigma_k)
        # swing contribution from ChipEV: swing_cev_coeff[k] * sum_cev_k
        # swing contribution from epsilon: swing_eps_coeff[k] * sum_eps_k
        for k in range(n_mult):
            n_k = counts_f[:, k]  # (B,)
            sqrt_n_k = np.sqrt(n_k)

            if sigma_cev[k] > 0 and np.abs(swing_cev_coeff[k]) > 1e-15:
                # sum_cev ~ Normal(n_k * mu, sqrt(n_k) * sigma)
                # But we want swing_cev_coeff * (sum_cev - n_k * mu_cev) since
                # the n_k * mu_cev part is already implicit in swing_const via
                # the EV formula. Actually: swing_const already accounts for the
                # deterministic EV at mean ChipEV. The stochastic part comes from
                # ChipEV deviating from mean.
                z_cev = rng.standard_normal(B)
                batch_swings += swing_cev_coeff[k] * sqrt_n_k * sigma_cev[k] * z_cev

            if sigma_eps[k] > 0 and np.abs(swing_eps_coeff[k]) > 1e-15:
                z_eps = rng.standard_normal(B)
                batch_swings += swing_eps_coeff[k] * sqrt_n_k * sigma_eps[k] * z_eps

        swings[sim_start:sim_end] = batch_swings

        if progress_callback is not None:
            progress_callback(sim_end, n_simulations)

    return swings


def _simulate_additive_eco_no_chip_ev_var(
    setup: Setup,
    sp: StructureParams,
    distributions: dict[int, StackDistributions],
    mean_chip_ev: dict[int, float],
    n_tourneys: int,
    n_simulations: int,
    ev_formula: EVFormula,
    rng: np.random.Generator,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Eco-mode additive simulation with fixed ChipEV (no ChipEV variance)."""
    n_mult = sp.n_mult
    buyin = setup.buyin

    probs = sp.probs.copy()
    probs /= probs.sum()

    coeffs_result = np.empty(n_mult, dtype=np.float64)
    consts_result = np.empty(n_mult, dtype=np.float64)
    for i in range(n_mult):
        stack = int(sp.stacks[i])
        part = sp.parts[i]
        coeffs_result[i] = buyin * part * (sp.w1[i] - sp.w23[i]) / (3.0 * stack)
        consts_result[i] = buyin * part * (sp.w1[i] / 3.0 + 2.0 * sp.w23[i] / 3.0 - 1.0)

    ev_const = np.empty(n_mult, dtype=np.float64)
    ev_coeff = np.empty(n_mult, dtype=np.float64)
    for k in range(n_mult):
        idx_arr = np.array([k], dtype=np.intp)
        ev_at_0 = float(compute_ev_additive_batch(
            idx_arr, np.array([0.0]), sp, ev_formula)[0])
        ev_at_1 = float(compute_ev_additive_batch(
            idx_arr, np.array([1.0]), sp, ev_formula)[0])
        ev_const[k] = ev_at_0
        ev_coeff[k] = ev_at_1 - ev_at_0

    swing_const = consts_result - ev_const
    swing_cev_coeff = coeffs_result - ev_coeff
    swing_eps_coeff = coeffs_result

    mu_cev, _, sigma_eps = _precompute_distribution_moments(
        sp, distributions, mean_chip_ev,
    )

    batch_size = min(n_simulations, 500_000)
    swings = np.empty(n_simulations, dtype=np.float64)

    for sim_start in range(0, n_simulations, batch_size):
        sim_end = min(sim_start + batch_size, n_simulations)
        B = sim_end - sim_start

        counts = rng.multinomial(n_tourneys, probs, size=B)  # (B, K)
        counts_f = counts.astype(np.float64)

        batch_swings = counts_f @ swing_const

        for k in range(n_mult):
            n_k = counts_f[:, k]
            sqrt_n_k = np.sqrt(n_k)

            if np.abs(swing_cev_coeff[k]) > 1e-15 and np.abs(mu_cev[k]) > 1e-15:
                # ChipEV is fixed at target mean, so only deterministic mean remains.
                batch_swings += swing_cev_coeff[k] * n_k * mu_cev[k]

            if sigma_eps[k] > 0 and np.abs(swing_eps_coeff[k]) > 1e-15:
                z_eps = rng.standard_normal(B)
                batch_swings += swing_eps_coeff[k] * sqrt_n_k * sigma_eps[k] * z_eps

        swings[sim_start:sim_end] = batch_swings

        if progress_callback is not None:
            progress_callback(sim_end, n_simulations)

    return swings


# ---------------------------------------------------------------------------
# Eco mode: CLT-based simulation (non-additive formulas)
# ---------------------------------------------------------------------------

def _simulate_full_detail_eco(
    sp: StructureParams,
    distributions: dict[int, StackDistributions],
    mean_chip_ev: dict[int, float],
    n_tourneys: int,
    n_simulations: int,
    ev_formula: EVFormula,
    rng: np.random.Generator,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Eco-mode non-additive simulation.

    Uses CLT for Result (same as additive eco for the Result side),
    and CLT-approximated per-multiplier mean ChipEV for the EV side.

    For AW EV: weighted mean ChipEV converges by CLT, so we sample
    per-multiplier mean ChipEV ~ Normal(mu_k, sigma_k / sqrt(n_k))
    and compute EV from those means.

    For AvgEV4: similar approach with theoretical counts and per-mult means.
    """
    probs = sp.probs.copy()
    probs /= probs.sum()

    mu_cev, sigma_cev, sigma_eps = _precompute_distribution_moments(
        sp, distributions, mean_chip_ev,
    )

    batch_size = min(n_simulations, 200_000)
    swings = np.empty(n_simulations, dtype=np.float64)

    for sim_start in range(0, n_simulations, batch_size):
        sim_end = min(sim_start + batch_size, n_simulations)
        B = sim_end - sim_start

        counts = rng.multinomial(n_tourneys, probs, size=B)  # (B, K)
        counts_f = counts.astype(np.float64)

        # --- Result side (same for all formulas) ---
        # Result_i = buyin * part * [((ChipNW+S)/(3S))*w1 + ((2S-ChipNW)/(3S))*w23 - 1]
        # sum(Result) = sum_k[ n_k * const_k + coeff_k * sum(ChipNW_k) ]
        # sum(ChipNW_k) = sum(ChipEV_k) + sum(eps_k)
        # CLT: sum(ChipEV_k) ~ Normal(n_k*mu_k, sqrt(n_k)*sigma_cev_k)
        #       sum(eps_k) ~ Normal(0, sqrt(n_k)*sigma_eps_k)
        result_consts = np.empty(sp.n_mult, dtype=np.float64)
        result_coeffs = np.empty(sp.n_mult, dtype=np.float64)
        for k in range(sp.n_mult):
            stack = float(sp.stacks[k])
            part = sp.parts[k]
            result_coeffs[k] = sp.buyin * part * (sp.w1[k] - sp.w23[k]) / (3.0 * stack)
            result_consts[k] = sp.buyin * part * (sp.w1[k] / 3.0 + 2.0 * sp.w23[k] / 3.0 - 1.0)

        # Deterministic result: n_k * const_k + coeff_k * n_k * mu_cev_k
        total_result = (counts_f @ result_consts
                        + (counts_f * mu_cev[np.newaxis, :]) @ result_coeffs)  # (B,)

        # Stochastic result
        for k in range(sp.n_mult):
            sqrt_n_k = np.sqrt(counts_f[:, k])
            if sigma_cev[k] > 0:
                total_result += result_coeffs[k] * sqrt_n_k * sigma_cev[k] * rng.standard_normal(B)
            if sigma_eps[k] > 0:
                total_result += result_coeffs[k] * sqrt_n_k * sigma_eps[k] * rng.standard_normal(B)

        # --- EV side ---
        # Sample per-multiplier mean ChipEV for each sim
        # mean_cev_k ~ Normal(mu_k, sigma_cev_k / sqrt(n_k))
        for b in range(B):
            per_mult_mean_cev = np.empty(sp.n_mult, dtype=np.float64)
            for k in range(sp.n_mult):
                n_k = int(counts[b, k])
                if n_k > 0 and sigma_cev[k] > 0:
                    per_mult_mean_cev[k] = mu_cev[k] + sigma_cev[k] / np.sqrt(n_k) * rng.standard_normal()
                else:
                    per_mult_mean_cev[k] = mu_cev[k]

            # Build synthetic per-mult arrays for compute_total_ev
            mult_indices = np.repeat(np.arange(sp.n_mult, dtype=np.intp), counts[b])
            chip_evs_synthetic = per_mult_mean_cev[mult_indices]

            total_ev = compute_total_ev(mult_indices, chip_evs_synthetic, sp, ev_formula)
            swings[sim_start + b] = total_result[b] - total_ev

        if progress_callback is not None:
            progress_callback(sim_end, n_simulations)

    return swings


# ---------------------------------------------------------------------------
# Precise mode: Full-detail simulation (non-additive formulas)
# ---------------------------------------------------------------------------

def _simulate_full_detail(
    sp: StructureParams,
    distributions: dict[int, StackDistributions],
    mean_chip_ev: dict[int, float],
    n_tourneys: int,
    n_simulations: int,
    ev_formula: EVFormula,
    rng: np.random.Generator,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Precise-mode: exact tournament-by-tournament sampling."""
    probs = sp.probs.copy()
    probs /= probs.sum()

    shifted_chip_ev: dict[int, DiscreteDistribution] = {}
    for stack_val in np.unique(sp.stacks):
        stack = int(stack_val)
        sd = distributions.get(stack)
        if sd is None:
            continue
        target = mean_chip_ev.get(stack, 0.0)
        shifted_chip_ev[stack] = sd.chip_ev.shifted(target - sd.chip_ev.mean)

    batch_size = _adaptive_batch_size(n_simulations, n_tourneys, 2_000)
    swings = np.empty(n_simulations, dtype=np.float64)

    for sim_start in range(0, n_simulations, batch_size):
        sim_end = min(sim_start + batch_size, n_simulations)
        B = sim_end - sim_start

        counts = rng.multinomial(n_tourneys, probs, size=B)

        flat_mult = np.empty(B * n_tourneys, dtype=np.intp)
        flat_cev = np.empty(B * n_tourneys, dtype=np.float64)
        flat_cnw = np.empty(B * n_tourneys, dtype=np.float64)

        cum_counts = np.zeros((B, sp.n_mult + 1), dtype=np.int64)
        for k in range(sp.n_mult):
            cum_counts[:, k + 1] = cum_counts[:, k] + counts[:, k]

        for k in range(sp.n_mult):
            total_k = int(counts[:, k].sum())
            if total_k == 0:
                continue

            stack = int(sp.stacks[k])
            sd = distributions.get(stack)
            cev_dist = shifted_chip_ev.get(stack)

            if sd is None or cev_dist is None:
                fallback = mean_chip_ev.get(stack, 0.0)
                all_cev = np.full(total_k, fallback)
                all_eps = np.zeros(total_k)
            else:
                all_cev = cev_dist.sample(total_k, rng)
                all_eps = _sample_allin_batch(all_cev, sd, rng)

            src_offset = 0
            for b in range(B):
                n_k = int(counts[b, k])
                if n_k == 0:
                    continue
                dst_start = b * n_tourneys + int(cum_counts[b, k])
                flat_mult[dst_start:dst_start + n_k] = k
                flat_cev[dst_start:dst_start + n_k] = all_cev[src_offset:src_offset + n_k]
                flat_cnw[dst_start:dst_start + n_k] = (
                    all_cev[src_offset:src_offset + n_k]
                    + all_eps[src_offset:src_offset + n_k]
                )
                src_offset += n_k

        all_results = compute_result_batch(flat_mult, flat_cnw, sp)
        result_sums = all_results.reshape(B, n_tourneys).sum(axis=1)

        for b in range(B):
            start = b * n_tourneys
            end = start + n_tourneys
            total_ev = compute_total_ev(
                flat_mult[start:end], flat_cev[start:end], sp, ev_formula,
            )
            swings[sim_start + b] = result_sums[b] - total_ev

        if progress_callback is not None:
            progress_callback(sim_end, n_simulations)

    return swings


# ---------------------------------------------------------------------------
# Precise mode: Additive simulation (exact sampling)
# ---------------------------------------------------------------------------

def _simulate_additive(
    setup: Setup,
    sp: StructureParams,
    distributions: dict[int, StackDistributions],
    mean_chip_ev: dict[int, float],
    n_tourneys: int,
    n_simulations: int,
    ev_formula: EVFormula,
    rng: np.random.Generator,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Precise-mode additive — exact sampling, vectorised per-multiplier."""
    buyin = setup.buyin
    n_mult = sp.n_mult

    probs = sp.probs.copy()
    probs /= probs.sum()

    coeffs_result = np.empty(n_mult, dtype=np.float64)
    consts_result = np.empty(n_mult, dtype=np.float64)
    for i in range(n_mult):
        stack = int(sp.stacks[i])
        part = sp.parts[i]
        coeffs_result[i] = buyin * part * (sp.w1[i] - sp.w23[i]) / (3.0 * stack)
        consts_result[i] = buyin * part * (sp.w1[i] / 3.0 + 2.0 * sp.w23[i] / 3.0 - 1.0)

    ev_const = np.empty(n_mult, dtype=np.float64)
    ev_coeff = np.empty(n_mult, dtype=np.float64)
    for k in range(n_mult):
        idx_arr = np.array([k], dtype=np.intp)
        ev_at_0 = float(compute_ev_additive_batch(
            idx_arr, np.array([0.0]), sp, ev_formula)[0])
        ev_at_1 = float(compute_ev_additive_batch(
            idx_arr, np.array([1.0]), sp, ev_formula)[0])
        ev_const[k] = ev_at_0
        ev_coeff[k] = ev_at_1 - ev_at_0

    swing_const = consts_result - ev_const
    swing_cev_coeff = coeffs_result - ev_coeff
    swing_eps_coeff = coeffs_result

    shifted_chip_ev: dict[int, DiscreteDistribution] = {}
    for stack in np.unique(sp.stacks):
        stack_int = int(stack)
        sd = distributions.get(stack_int)
        if sd is None:
            continue
        target = mean_chip_ev.get(stack_int, 0.0)
        shifted_chip_ev[stack_int] = sd.chip_ev.shifted(target - sd.chip_ev.mean)

    batch_size = _adaptive_batch_size(n_simulations, n_tourneys, 50_000)
    swings = np.empty(n_simulations, dtype=np.float64)

    for sim_start in range(0, n_simulations, batch_size):
        sim_end = min(sim_start + batch_size, n_simulations)
        B = sim_end - sim_start

        counts = rng.multinomial(n_tourneys, probs, size=B)
        batch_swings = counts.astype(np.float64) @ swing_const

        for k in range(n_mult):
            m_counts = counts[:, k]
            stack = int(sp.stacks[k])
            sd = distributions.get(stack)
            cev_dist = shifted_chip_ev.get(stack)
            if sd is None or cev_dist is None:
                continue

            active_mask = m_counts > 0
            if not active_mask.any():
                continue

            active_counts = m_counts[active_mask]
            total_n = int(active_counts.sum())

            all_cev = cev_dist.sample(total_n, rng)
            all_eps = _sample_allin_batch(all_cev, sd, rng)

            sum_cev = _segmented_sum_via_cumsum(all_cev, active_counts)
            sum_eps = _segmented_sum_via_cumsum(all_eps, active_counts)

            batch_swings[active_mask] += (
                swing_cev_coeff[k] * sum_cev
                + swing_eps_coeff[k] * sum_eps
            )

        swings[sim_start:sim_end] = batch_swings

        if progress_callback is not None:
            progress_callback(sim_end, n_simulations)

    return swings


# ---------------------------------------------------------------------------
# Threading wrapper for precise mode
# ---------------------------------------------------------------------------

def _get_n_threads() -> int:
    cpu = os.cpu_count() or 4
    return max(1, min(cpu, 8))


def _run_precise_threaded(
    worker_fn,
    worker_args: tuple,
    n_simulations: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Split simulations across threads for precise mode.

    Each thread gets its own RNG (spawned from the parent) and computes
    a chunk of simulations independently.
    """
    n_threads = _get_n_threads()
    if n_threads <= 1 or n_simulations < 10_000:
        return worker_fn(*worker_args, n_simulations,
                         np.random.default_rng(), progress_callback)

    parent_rng = np.random.default_rng()
    child_rngs = [np.random.default_rng(s)
                  for s in parent_rng.spawn(n_threads)]

    chunk_sizes = [n_simulations // n_threads] * n_threads
    for i in range(n_simulations % n_threads):
        chunk_sizes[i] += 1

    completed = [0]

    def _thread_progress(done: int, total: int) -> None:
        pass  # per-thread progress not aggregated

    swings = np.empty(n_simulations, dtype=np.float64)
    futures = {}

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        offset = 0
        for t in range(n_threads):
            sz = chunk_sizes[t]
            if sz == 0:
                continue
            future = pool.submit(
                worker_fn, *worker_args, sz, child_rngs[t], _thread_progress,
            )
            futures[future] = (offset, offset + sz)
            offset += sz

        done_count = 0
        for future in as_completed(futures):
            start, end = futures[future]
            swings[start:end] = future.result()
            done_count += end - start
            if progress_callback is not None:
                progress_callback(done_count, n_simulations)

    return swings


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def simulate_pool(
    setup: Setup,
    participations: list[float],
    mean_chip_ev: dict[int, float],
    distributions: dict[int, StackDistributions],
    n_tourneys: int,
    n_simulations: int,
    rng: np.random.Generator | None = None,
    ev_formula: EVFormula = EVFormula.AVG_EV,
    progress_callback: Callable[[int, int], None] | None = None,
    mode: str = "precise",
    target: str = "swing",
    mean_chip_ev_per_multiplier: list[float] | None = None,
) -> SimulationResult:
    """Run Monte Carlo simulation.

    Parameters
    ----------
    mode : str
        "eco" for CLT-based fast simulation,
        "no_chip_ev_var" for CLT simulation with fixed ChipEV mean and no
        ChipEV variance,
        "precise" for exact sampling.
    target : str
        "swing" (Result - EV) or "ev" (EV only) or "result" (Result only).
    mean_chip_ev_per_multiplier : list[float] | None
        Optional per-multiplier target mean ChipEV (len = number of multipliers).
        Used for EV/Result targets when multiple multipliers share the same stack size.
    """
    if rng is None:
        rng = np.random.default_rng()

    sp = build_structure_params(setup, participations)

    def _get_target_means_per_k() -> np.ndarray:
        """Return array mu_k of length K (target mean ChipEV per multiplier index)."""
        K = sp.n_mult
        mu = np.zeros(K, dtype=np.float64)
        if mean_chip_ev_per_multiplier is not None and len(mean_chip_ev_per_multiplier) == K:
            mu[:] = np.asarray(mean_chip_ev_per_multiplier, dtype=np.float64)
            return mu
        for k in range(K):
            stack = int(sp.stacks[k])
            mu[k] = float(mean_chip_ev.get(stack, 0.0))
        return mu

    def _get_sigma_cev_per_k() -> np.ndarray:
        """Return array sigma_k of length K (ChipEV std per multiplier index)."""
        K = sp.n_mult
        sigma = np.zeros(K, dtype=np.float64)
        for k in range(K):
            stack = int(sp.stacks[k])
            sd = distributions.get(stack)
            if sd is not None:
                sigma[k] = float(sd.chip_ev.std)
        return sigma

    def _compute_total_ev_from_counts_means(
        counts_row: np.ndarray,
        mean_cev_row: np.ndarray,
        formula: EVFormula,
    ) -> float:
        """Compute total EV without per-tourney arrays (eco helpers)."""
        n = int(counts_row.sum())
        if n <= 0:
            return 0.0

        if formula == EVFormula.AW_EV:
            # Normalize ChipEV to min_stack if variable stacks
            if sp.has_variable_stacks and sp.min_stack > 0:
                mean_norm = mean_cev_row * (float(sp.min_stack) / sp.stacks.astype(np.float64))
            else:
                mean_norm = mean_cev_row

            total_adj = sp.parts * (sp.w1 - sp.w23)
            prob_adj = sp.probs * sp.parts
            mask_4plus = sp.mult_values >= 4
            if mask_4plus.any() and prob_adj[mask_4plus].sum() > 0:
                w4plus = float(np.dot(total_adj[mask_4plus], prob_adj[mask_4plus])
                               / prob_adj[mask_4plus].sum())
            else:
                w4plus = 0.0

            weights = np.where(sp.mult_values < 4, sp.mult_values, w4plus).astype(np.float64)

            num = float(np.dot(mean_norm * weights, counts_row))
            den = float(np.dot(weights, counts_row))
            if den == 0:
                return 0.0
            wcev = num / den

            S = float(sp.min_stack) if (sp.min_stack > 0) else float(sp.avg_stack)
            if S == 0:
                return 0.0
            return float(
                sp.buyin * n * (
                    (wcev + S) / (3.0 * S) * sp.w1_avg
                    + (2.0 * S - wcev) / (3.0 * S) * sp.w23_avg
                    - 1.0
                )
            )

        if formula == EVFormula.AVG_EV_4:
            total = 0.0
            for k in range(sp.n_mult):
                theo_count = n * sp.probs[k]
                if theo_count == 0:
                    continue
                mv = sp.mult_values[k]
                mean_k = float(mean_cev_row[k]) if counts_row[k] > 0 else 0.0
                if mv < 4:
                    S = float(sp.stacks[k])
                    if S == 0:
                        continue
                    part = float(sp.parts[k])
                    ev_k = sp.buyin * part * (
                        (mean_k + S) / (3.0 * S) * sp.w1[k]
                        + (2.0 * S - mean_k) / (3.0 * S) * sp.w23[k]
                        - 1.0
                    ) * theo_count
                else:
                    S = float(sp.avg_stack_4plus) if sp.avg_stack_4plus > 0 else float(sp.stacks[k])
                    if S == 0:
                        continue
                    ev_k = sp.buyin * (
                        (mean_k + S) / (3.0 * S) * sp.w1_avg_4plus
                        + (2.0 * S - mean_k) / (3.0 * S) * sp.w23_avg_4plus
                        - 1.0
                    ) * theo_count
                total += ev_k
            return float(total)

        # Fallback: build synthetic arrays (shouldn't happen for eco paths below)
        mult_indices = np.repeat(np.arange(sp.n_mult, dtype=np.intp), counts_row.astype(int))
        chip_evs_synth = np.repeat(mean_cev_row, counts_row.astype(int))
        return float(compute_total_ev(mult_indices, chip_evs_synth, sp, formula))

    # ------------------------------------------------------------------
    # Simulation kernels that can optionally return Result/EV separately
    # ------------------------------------------------------------------
    def _simulate_additive_components_precise(
        n_sims: int,
        rng_t: np.random.Generator,
        pcb: Callable[[int, int], None] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (result_totals, ev_totals) for additive formulas (precise)."""
        buyin = setup.buyin
        n_mult = sp.n_mult

        probs = sp.probs.copy()
        probs /= probs.sum()

        coeffs_result = np.empty(n_mult, dtype=np.float64)
        consts_result = np.empty(n_mult, dtype=np.float64)
        for i in range(n_mult):
            stack = int(sp.stacks[i])
            part = sp.parts[i]
            coeffs_result[i] = buyin * part * (sp.w1[i] - sp.w23[i]) / (3.0 * stack)
            consts_result[i] = buyin * part * (sp.w1[i] / 3.0 + 2.0 * sp.w23[i] / 3.0 - 1.0)

        ev_const = np.empty(n_mult, dtype=np.float64)
        ev_coeff = np.empty(n_mult, dtype=np.float64)
        for k in range(n_mult):
            idx_arr = np.array([k], dtype=np.intp)
            ev_at_0 = float(compute_ev_additive_batch(
                idx_arr, np.array([0.0]), sp, ev_formula)[0])
            ev_at_1 = float(compute_ev_additive_batch(
                idx_arr, np.array([1.0]), sp, ev_formula)[0])
            ev_const[k] = ev_at_0
            ev_coeff[k] = ev_at_1 - ev_at_0

        shifted_chip_ev: dict[int, DiscreteDistribution] = {}
        shifted_chip_ev_by_k: dict[int, DiscreteDistribution] = {}
        for k in range(n_mult):
            stack_int = int(sp.stacks[k])
            sd = distributions.get(stack_int)
            if sd is None:
                continue
            if mean_chip_ev_per_multiplier is not None and k < len(mean_chip_ev_per_multiplier):
                target_mean = float(mean_chip_ev_per_multiplier[k])
            else:
                target_mean = float(mean_chip_ev.get(stack_int, 0.0))
            shifted_chip_ev_by_k[k] = sd.chip_ev.shifted(target_mean - sd.chip_ev.mean)

        batch_size = _adaptive_batch_size(n_sims, n_tourneys, 50_000)
        result_totals = np.empty(n_sims, dtype=np.float64)
        ev_totals = np.empty(n_sims, dtype=np.float64)

        for sim_start in range(0, n_sims, batch_size):
            sim_end = min(sim_start + batch_size, n_sims)
            B = sim_end - sim_start

            counts = rng_t.multinomial(n_tourneys, probs, size=B)
            counts_f = counts.astype(np.float64)

            # Start with deterministic parts
            batch_result = counts_f @ consts_result
            batch_ev = counts_f @ ev_const

            for k in range(n_mult):
                m_counts = counts[:, k]
                stack = int(sp.stacks[k])
                sd = distributions.get(stack)
                cev_dist = shifted_chip_ev_by_k.get(k)
                if sd is None or cev_dist is None:
                    # Fallback: fixed mean, no variance
                    if mean_chip_ev_per_multiplier is not None and k < len(mean_chip_ev_per_multiplier):
                        fallback_mean = float(mean_chip_ev_per_multiplier[k])
                    else:
                        fallback_mean = float(mean_chip_ev.get(stack, 0.0))
                    if ev_coeff[k] != 0.0:
                        batch_ev += ev_coeff[k] * counts_f[:, k] * fallback_mean
                    if coeffs_result[k] != 0.0:
                        batch_result += coeffs_result[k] * counts_f[:, k] * fallback_mean
                    continue

                active_mask = m_counts > 0
                if not active_mask.any():
                    continue

                active_counts = m_counts[active_mask]
                total_n = int(active_counts.sum())

                all_cev = cev_dist.sample(total_n, rng_t)
                all_eps = _sample_allin_batch(all_cev, sd, rng_t)

                sum_cev = _segmented_sum_via_cumsum(all_cev, active_counts)
                sum_eps = _segmented_sum_via_cumsum(all_eps, active_counts)

                batch_ev[active_mask] += ev_coeff[k] * sum_cev
                batch_result[active_mask] += coeffs_result[k] * (sum_cev + sum_eps)

            result_totals[sim_start:sim_end] = batch_result
            ev_totals[sim_start:sim_end] = batch_ev

            if pcb is not None:
                pcb(sim_end, n_sims)

        return result_totals, ev_totals

    def _simulate_full_detail_components_precise(
        n_sims: int,
        rng_t: np.random.Generator,
        pcb: Callable[[int, int], None] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (result_totals, ev_totals) for non-additive formulas (precise)."""
        probs = sp.probs.copy()
        probs /= probs.sum()

        shifted_chip_ev_by_k: dict[int, DiscreteDistribution] = {}
        for k in range(sp.n_mult):
            stack = int(sp.stacks[k])
            sd = distributions.get(stack)
            if sd is None:
                continue
            if mean_chip_ev_per_multiplier is not None and k < len(mean_chip_ev_per_multiplier):
                target_mean = float(mean_chip_ev_per_multiplier[k])
            else:
                target_mean = float(mean_chip_ev.get(stack, 0.0))
            shifted_chip_ev_by_k[k] = sd.chip_ev.shifted(target_mean - sd.chip_ev.mean)

        batch_size = _adaptive_batch_size(n_sims, n_tourneys, 2_000)
        result_totals = np.empty(n_sims, dtype=np.float64)
        ev_totals = np.empty(n_sims, dtype=np.float64)

        for sim_start in range(0, n_sims, batch_size):
            sim_end = min(sim_start + batch_size, n_sims)
            B = sim_end - sim_start

            counts = rng_t.multinomial(n_tourneys, probs, size=B)

            flat_mult = np.empty(B * n_tourneys, dtype=np.intp)
            flat_cev = np.empty(B * n_tourneys, dtype=np.float64)
            flat_cnw = np.empty(B * n_tourneys, dtype=np.float64)

            cum_counts = np.zeros((B, sp.n_mult + 1), dtype=np.int64)
            for k in range(sp.n_mult):
                cum_counts[:, k + 1] = cum_counts[:, k] + counts[:, k]

            for k in range(sp.n_mult):
                total_k = int(counts[:, k].sum())
                if total_k == 0:
                    continue

                stack = int(sp.stacks[k])
                sd = distributions.get(stack)
                cev_dist = shifted_chip_ev_by_k.get(k)

                if sd is None or cev_dist is None:
                    if mean_chip_ev_per_multiplier is not None and k < len(mean_chip_ev_per_multiplier):
                        fallback = float(mean_chip_ev_per_multiplier[k])
                    else:
                        fallback = float(mean_chip_ev.get(stack, 0.0))
                    all_cev = np.full(total_k, fallback)
                    all_eps = np.zeros(total_k)
                else:
                    all_cev = cev_dist.sample(total_k, rng_t)
                    all_eps = _sample_allin_batch(all_cev, sd, rng_t)

                src_offset = 0
                for b in range(B):
                    n_k = int(counts[b, k])
                    if n_k == 0:
                        continue
                    dst_start = b * n_tourneys + int(cum_counts[b, k])
                    flat_mult[dst_start:dst_start + n_k] = k
                    flat_cev[dst_start:dst_start + n_k] = all_cev[src_offset:src_offset + n_k]
                    flat_cnw[dst_start:dst_start + n_k] = (
                        all_cev[src_offset:src_offset + n_k]
                        + all_eps[src_offset:src_offset + n_k]
                    )
                    src_offset += n_k

            all_results = compute_result_batch(flat_mult, flat_cnw, sp)
            result_sums = all_results.reshape(B, n_tourneys).sum(axis=1)

            for b in range(B):
                start = b * n_tourneys
                end = start + n_tourneys
                total_ev = compute_total_ev(
                    flat_mult[start:end], flat_cev[start:end], sp, ev_formula,
                )
                result_totals[sim_start + b] = result_sums[b]
                ev_totals[sim_start + b] = total_ev

            if pcb is not None:
                pcb(sim_end, n_sims)

        return result_totals, ev_totals

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------
    if mode == "eco":
        if target == "swing":
            if is_additive(ev_formula):
                swings = _simulate_additive_eco(
                    setup, sp, distributions, mean_chip_ev,
                    n_tourneys, n_simulations, ev_formula, rng, progress_callback,
                )
            else:
                # Non-additive formulas: use AvgEV as additive proxy in eco mode.
                swings = _simulate_additive_eco(
                    setup, sp, distributions, mean_chip_ev,
                    n_tourneys, n_simulations, EVFormula.AVG_EV, rng, progress_callback,
                )
        else:
            # Eco mode for EV/Result: simulate counts and ChipEV aggregates only.
            probs = sp.probs.copy()
            probs /= probs.sum()
            mu_k = _get_target_means_per_k()
            sigma_k = _get_sigma_cev_per_k()

            batch_size = min(n_simulations, 200_000 if is_additive(ev_formula) else 50_000)
            ev_totals = np.empty(n_simulations, dtype=np.float64)
            result_totals = np.empty(n_simulations, dtype=np.float64) if target == "result" else None

            # Precompute EV linear coefficients for additive formulas: EV = const + coeff*ChipEV
            ev_const = np.empty(sp.n_mult, dtype=np.float64)
            ev_coeff = np.empty(sp.n_mult, dtype=np.float64)
            for k in range(sp.n_mult):
                idx_arr = np.array([k], dtype=np.intp)
                ev_at_0 = float(compute_ev_additive_batch(idx_arr, np.array([0.0]), sp, ev_formula)[0])
                ev_at_1 = float(compute_ev_additive_batch(idx_arr, np.array([1.0]), sp, ev_formula)[0])
                ev_const[k] = ev_at_0
                ev_coeff[k] = ev_at_1 - ev_at_0

            # Precompute Result linear coefficients: Result = const + coeff*ChipNW
            if target == "result":
                coeff_res = np.empty(sp.n_mult, dtype=np.float64)
                const_res = np.empty(sp.n_mult, dtype=np.float64)
                for k in range(sp.n_mult):
                    stack = float(sp.stacks[k]) if sp.stacks[k] != 0 else 1.0
                    part = float(sp.parts[k])
                    coeff_res[k] = sp.buyin * part * (sp.w1[k] - sp.w23[k]) / (3.0 * stack)
                    const_res[k] = sp.buyin * part * (sp.w1[k] / 3.0 + 2.0 * sp.w23[k] / 3.0 - 1.0)

            for sim_start in range(0, n_simulations, batch_size):
                sim_end = min(sim_start + batch_size, n_simulations)
                B = sim_end - sim_start

                counts = rng.multinomial(n_tourneys, probs, size=B)  # (B,K)
                counts_f = counts.astype(np.float64)

                if is_additive(ev_formula):
                    # Sum ChipEV per k: Normal(n_k*mu, sqrt(n_k)*sigma)
                    sum_cev = counts_f * mu_k[np.newaxis, :]
                    if (sigma_k > 0).any():
                        sum_cev += np.sqrt(counts_f) * sigma_k[np.newaxis, :] * rng.standard_normal(size=(B, sp.n_mult))
                    total_ev = counts_f @ ev_const + (sum_cev @ ev_coeff)
                else:
                    # Sample mean ChipEV per k: Normal(mu, sigma/sqrt(n_k))
                    mean_cev = np.tile(mu_k, (B, 1))
                    for k in range(sp.n_mult):
                        if sigma_k[k] <= 0:
                            continue
                        n_k = counts_f[:, k]
                        mask = n_k > 0
                        if mask.any():
                            mean_cev[mask, k] = mu_k[k] + sigma_k[k] / np.sqrt(n_k[mask]) * rng.standard_normal(mask.sum())
                    total_ev = np.empty(B, dtype=np.float64)
                    for b in range(B):
                        total_ev[b] = _compute_total_ev_from_counts_means(counts_f[b], mean_cev[b], ev_formula)

                ev_totals[sim_start:sim_end] = total_ev

                if target == "result" and result_totals is not None:
                    # For Result we ignore all-in luck in eco mode; approximate ChipNW by ChipEV only.
                    total_res = counts_f @ const_res + (sum_cev @ coeff_res)  # type: ignore[name-defined]
                    result_totals[sim_start:sim_end] = total_res

                if progress_callback is not None:
                    progress_callback(sim_end, n_simulations)

            swings = None
    elif mode == "no_chip_ev_var":
        if target == "swing":
            if is_additive(ev_formula):
                swings = _simulate_additive_eco_no_chip_ev_var(
                    setup, sp, distributions, mean_chip_ev,
                    n_tourneys, n_simulations, ev_formula, rng, progress_callback,
                )
            else:
                swings = _simulate_additive_eco_no_chip_ev_var(
                    setup, sp, distributions, mean_chip_ev,
                    n_tourneys, n_simulations, EVFormula.AVG_EV, rng, progress_callback,
                )
        else:
            # Fixed ChipEV: simulate multiplier counts only; ChipEV equals target mean per multiplier.
            probs = sp.probs.copy()
            probs /= probs.sum()
            mu_k = _get_target_means_per_k()

            batch_size = min(n_simulations, 500_000)
            ev_totals = np.empty(n_simulations, dtype=np.float64)
            result_totals = np.empty(n_simulations, dtype=np.float64) if target == "result" else None

            ev_const = np.empty(sp.n_mult, dtype=np.float64)
            ev_coeff = np.empty(sp.n_mult, dtype=np.float64)
            for k in range(sp.n_mult):
                idx_arr = np.array([k], dtype=np.intp)
                ev_at_0 = float(compute_ev_additive_batch(idx_arr, np.array([0.0]), sp, ev_formula)[0])
                ev_at_1 = float(compute_ev_additive_batch(idx_arr, np.array([1.0]), sp, ev_formula)[0])
                ev_const[k] = ev_at_0
                ev_coeff[k] = ev_at_1 - ev_at_0

            if target == "result":
                coeff_res = np.empty(sp.n_mult, dtype=np.float64)
                const_res = np.empty(sp.n_mult, dtype=np.float64)
                for k in range(sp.n_mult):
                    stack = float(sp.stacks[k]) if sp.stacks[k] != 0 else 1.0
                    part = float(sp.parts[k])
                    coeff_res[k] = sp.buyin * part * (sp.w1[k] - sp.w23[k]) / (3.0 * stack)
                    const_res[k] = sp.buyin * part * (sp.w1[k] / 3.0 + 2.0 * sp.w23[k] / 3.0 - 1.0)

            for sim_start in range(0, n_simulations, batch_size):
                sim_end = min(sim_start + batch_size, n_simulations)
                B = sim_end - sim_start
                counts = rng.multinomial(n_tourneys, probs, size=B)
                counts_f = counts.astype(np.float64)

                if is_additive(ev_formula):
                    total_ev = counts_f @ (ev_const + ev_coeff * mu_k)
                else:
                    total_ev = np.empty(B, dtype=np.float64)
                    for b in range(B):
                        total_ev[b] = _compute_total_ev_from_counts_means(counts_f[b], mu_k, ev_formula)

                ev_totals[sim_start:sim_end] = total_ev

                if target == "result" and result_totals is not None:
                    total_res = counts_f @ (const_res + coeff_res * mu_k)  # type: ignore[name-defined]
                    result_totals[sim_start:sim_end] = total_res

                if progress_callback is not None:
                    progress_callback(sim_end, n_simulations)

            swings = None

    if mode == "precise":
        if is_additive(ev_formula):
            def _worker(n_sims, rng_t, pcb):
                return _simulate_additive_components_precise(n_sims, rng_t, pcb)

            # Reuse threading wrapper by packing tuple into object array.
            # (ThreadPoolExecutor expects a single ndarray return, so we run single-threaded here
            #  when target requires components.)
            if target == "swing":
                def _add_worker(n_sims, rng_t, pcb):
                    return _simulate_additive(
                        setup, sp, distributions, mean_chip_ev,
                        n_tourneys, n_sims, ev_formula, rng_t, pcb,
                    )
                swings = _run_precise_threaded(
                    _add_worker, (), n_simulations, progress_callback,
                )
                result_totals = None
                ev_totals = None
            else:
                # For EV/Result targets we need components; run single-threaded to keep memory predictable.
                result_totals, ev_totals = _simulate_additive_components_precise(
                    n_simulations, rng, progress_callback,
                )
                swings = None
        else:
            if target == "swing":
                def _fd_worker(n_sims, rng_t, pcb):
                    return _simulate_full_detail(
                        sp, distributions, mean_chip_ev,
                        n_tourneys, n_sims, ev_formula, rng_t, pcb,
                    )
                swings = _run_precise_threaded(
                    _fd_worker, (), n_simulations, progress_callback,
                )
                result_totals = None
                ev_totals = None
            else:
                result_totals, ev_totals = _simulate_full_detail_components_precise(
                    n_simulations, rng, progress_callback,
                )
                swings = None

    # Select output series
    metric_label = "Swing"
    if target == "swing":
        assert swings is not None
        metric = swings
        metric_label = "Swing"
    elif target == "ev":
        assert ev_totals is not None
        metric = ev_totals
        metric_label = "EV"
    elif target == "result":
        assert result_totals is not None
        metric = result_totals
        metric_label = "Result"
    else:
        assert swings is not None
        metric = swings
        metric_label = "Swing"

    sigma_analytical = 0.0
    sigma_analytical_portfolio = 0.0
    if target == "swing":
        sigma_analytical = compute_analytical_sigma(
            setup, participations, mean_chip_ev, distributions,
            ev_formula=ev_formula,
        )
        sigma_analytical_portfolio = sigma_analytical * np.sqrt(n_tourneys)

    simulated_std_portfolio = float(np.std(metric))
    simulated_std_per_tourney = simulated_std_portfolio / np.sqrt(n_tourneys)

    pct_keys = [1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0]
    pct_vals = np.percentile(metric, pct_keys)
    percentiles = dict(zip(pct_keys, pct_vals.tolist()))

    ci_95 = (float(np.percentile(metric, 2.5)), float(np.percentile(metric, 97.5)))
    ci_99 = (float(np.percentile(metric, 0.5)), float(np.percentile(metric, 99.5)))

    return SimulationResult(
        swings=metric,
        metric_label=metric_label,
        sigma_analytical=sigma_analytical,
        sigma_analytical_portfolio=sigma_analytical_portfolio,
        sigma_simulated=simulated_std_per_tourney,
        sigma_simulated_portfolio=simulated_std_portfolio,
        ci_95=ci_95,
        ci_99=ci_99,
        percentiles=percentiles,
        prob_positive=float(np.mean(metric > 0)),
        prob_negative=float(np.mean(metric < 0)),
        mean_swing=float(np.mean(metric)),
        n_tourneys=n_tourneys,
        n_simulations=n_simulations,
        config_name=setup.name,
        buyin=setup.buyin,
        poker_room=setup.poker_room,
        reservation=setup.reservation,
        buyin_native=setup.buyin,
        ev_formula_name=ev_formula.value,
    )

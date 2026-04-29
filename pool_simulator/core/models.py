"""Data models for pool simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Multiplier:
    """Single multiplier tier within a tournament structure."""

    item_index: int
    multiplier_value: float
    approx_multiplier: float
    w1: float
    w2: float
    w3: float
    w23: float
    probability: float
    frequency: float
    default_participation: float
    max_participation: float
    min_participation: float
    stack_size: int


@dataclass
class Setup:
    """Tournament setup — one specific room + limit + type combination."""

    tourney_id: int
    name: str
    poker_room: str
    reservation: str
    buyin: float
    tournament_type: str
    default_stack: int
    has_variable_stacks: bool
    unique_stacks: list[int]
    structure_id: int
    multipliers: list[Multiplier]


class DiscreteDistribution:
    """Discretized probability distribution with efficient sampling."""

    __slots__ = ("values", "probabilities", "_alias_prob", "_alias_idx", "_n")

    def __init__(self, values: np.ndarray, probabilities: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=np.float64)
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self._n = len(self.values)
        self._alias_prob: np.ndarray | None = None
        self._alias_idx: np.ndarray | None = None

    def _build_alias_table(self) -> None:
        """Build Vose alias table for O(1) sampling."""
        n = self._n
        prob = self.probabilities * n
        alias_prob = np.zeros(n, dtype=np.float64)
        alias_idx = np.zeros(n, dtype=np.int64)

        small: list[int] = []
        large: list[int] = []
        for i in range(n):
            if prob[i] < 1.0:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            s = small.pop()
            l = large.pop()
            alias_prob[s] = prob[s]
            alias_idx[s] = l
            prob[l] = prob[l] + prob[s] - 1.0
            if prob[l] < 1.0:
                small.append(l)
            else:
                large.append(l)

        while large:
            alias_prob[large.pop()] = 1.0
        while small:
            alias_prob[small.pop()] = 1.0

        self._alias_prob = alias_prob
        self._alias_idx = alias_idx

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw *n* samples using the alias method — O(n) time, O(1) per sample."""
        if self._alias_prob is None:
            self._build_alias_table()
        assert self._alias_prob is not None and self._alias_idx is not None

        col = rng.integers(0, self._n, size=n)
        coin = rng.random(size=n)
        idx = np.where(coin < self._alias_prob[col], col, self._alias_idx[col])
        return self.values[idx]

    def shifted(self, delta: float) -> DiscreteDistribution:
        """Return a new distribution with values shifted by *delta*."""
        return DiscreteDistribution(self.values + delta, self.probabilities.copy())

    @property
    def mean(self) -> float:
        return float(np.dot(self.values, self.probabilities))

    @property
    def std(self) -> float:
        mu = self.mean
        return float(np.sqrt(np.dot((self.values - mu) ** 2, self.probabilities)))


@dataclass
class ConditionalBin:
    """All-in luck distribution conditioned on a ChipEV range."""

    ev_range: tuple[float, float]
    distribution: DiscreteDistribution
    n_observations: int
    std: float
    original_mean: float


@dataclass
class StackDistributions:
    """All distributions for a given stack size."""

    stack_size: int
    n_total: int
    chip_ev: DiscreteDistribution
    chip_ev_stats: dict[str, float]
    allin_unconditional: DiscreteDistribution
    allin_stats_unconditional: dict[str, float]
    conditional_bins: dict[int, ConditionalBin]
    conditional_bin_keys: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        if len(self.conditional_bins) > 0 and len(self.conditional_bin_keys) == 0:
            self.conditional_bin_keys = np.array(
                sorted(self.conditional_bins.keys()), dtype=np.float64
            )

    def find_conditional_bin(self, chip_ev: float) -> ConditionalBin | None:
        if len(self.conditional_bin_keys) == 0:
            return None
        idx = int(np.argmin(np.abs(self.conditional_bin_keys - chip_ev)))
        key = int(self.conditional_bin_keys[idx])
        return self.conditional_bins.get(key)


@dataclass
class MeanChipEVEntry:
    """Mean ChipEV record for a setup or setup × stack combination."""

    tourney_id: int
    mean_chip_ev: float
    n_observations: int
    stack_size: int | None = None
    median_chip_ev: float | None = None
    std_chip_ev: float | None = None
    has_variable_stacks: bool = False


@dataclass
class MeanChipEVStore:
    """Lookup store for default mean ChipEV values."""

    overall: dict[int, MeanChipEVEntry]
    by_stack: dict[str, MeanChipEVEntry]

    MIN_OBSERVATIONS = 130

    def get_default(self, setup: Setup) -> dict[int, float]:
        """Return {stack_size: mean_chip_ev} for a setup.

        For variable-stack setups, returns one entry per unique stack.
        Entries with fewer than MIN_OBSERVATIONS are treated as unreliable
        and marked 0.0 for subsequent proportional scaling by the caller.
        """
        result: dict[int, float] = {}
        if setup.has_variable_stacks:
            for stack in setup.unique_stacks:
                key = f"{setup.tourney_id}_{stack}"
                entry = self.by_stack.get(key)
                if entry is not None and entry.n_observations >= self.MIN_OBSERVATIONS:
                    result[stack] = entry.mean_chip_ev
                else:
                    result[stack] = 0.0
        else:
            entry = self.overall.get(setup.tourney_id)
            if entry is not None:
                result[setup.default_stack] = entry.mean_chip_ev
            else:
                result[setup.default_stack] = 0.0
        return result


@dataclass
class SimulationResult:
    """Output of a Monte Carlo simulation run."""

    swings: np.ndarray

    # Analytical (from whitepaper formula: Var_A + Var_B)
    sigma_analytical: float              # per-tournament, dollars
    sigma_analytical_portfolio: float    # portfolio of N, dollars

    # Simulated (from actual MC runs)
    sigma_simulated: float               # per-tournament (= portfolio / sqrt(N)), dollars
    sigma_simulated_portfolio: float     # std(swings), dollars

    ci_95: tuple[float, float]
    ci_99: tuple[float, float]
    percentiles: dict[float, float]
    prob_positive: float
    prob_negative: float
    mean_swing: float
    n_tourneys: int
    n_simulations: int

    config_name: str = ""
    buyin: float = 1.0
    poker_room: str = ""
    reservation: str = ""
    buyin_native: float = 0.0
    ev_formula_name: str = ""
    metric_label: str = "Swing"


EUR_ROOMS: dict[str, set[str]] = {
    "Betclic": set(),
    "Winamax": set(),
    "Winamax (old)": set(),
    "Winamax (new)": set(),
    "iPoker": set(),
    "Poker Stars": {"IT", "FR", "ES"},
}


def is_eur_room(poker_room: str, reservation: str = "") -> bool:
    """Return True if this room/reservation uses EUR as native currency."""
    if poker_room in ("Betclic", "Winamax", "Winamax (old)", "Winamax (new)", "iPoker"):
        return True
    if poker_room == "Poker Stars" and reservation in ("IT", "FR", "ES"):
        return True
    return False


@dataclass
class PortfolioEntry:
    """One setup configuration within a portfolio."""

    name: str
    setup: Setup
    participations: list[float]
    chip_evs: dict[int, float]
    ev_formula_name: str
    n_tourneys: int
    eur_rate: float = 1.0
    chip_evs_per_multiplier: list[float] = field(default_factory=list)


@dataclass
class PortfolioEntryResult:
    """Per-entry simulation result within a portfolio (all in USD)."""

    name: str
    n_tourneys: int
    buyin_usd: float
    sigma_per_tourney_usd: float
    sigma_portfolio_usd: float
    sigma_analytical_usd: float
    sigma_analytical_portfolio_usd: float
    swings_usd: np.ndarray
    poker_room: str = ""
    reservation: str = ""
    ev_formula_name: str = ""
    buyin_native: float = 0.0


@dataclass
class PortfolioResult:
    """Combined portfolio simulation result."""

    entry_results: list[PortfolioEntryResult]
    combined: SimulationResult
    total_n_tourneys: int

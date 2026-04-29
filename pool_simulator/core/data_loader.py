"""Load and parse processed data files into domain models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pool_simulator.core.models import (
    ConditionalBin,
    DiscreteDistribution,
    MeanChipEVEntry,
    MeanChipEVStore,
    Multiplier,
    Setup,
    StackDistributions,
)
from pool_simulator.core.paths import get_data_dir

DEFAULT_DATA_DIR = get_data_dir()


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fix_stack_sizes(
    multipliers: list[Multiplier],
    room: str,
    reservation: str,
    tournament_type: str,
) -> None:
    """Override stack_size per multiplier based on known room rules.

    GGpoker:           x2,x3=300; x4,x5=500; x10=500; x50,x100=800; jackpot=1000
    Poker Stars IT/COM: x2,x3=300; x4,x5=400; x10+=500
    Poker Stars other regular: all=500
    Poker Stars other flash:   all=300
    iPoker ES/FR flash: all=200
    Winamax (old/new) nitro:   all=300
    Winamax (old/new) regular: all=500
    """
    if room == "GGpoker":
        for m in multipliers:
            mv = m.multiplier_value
            if mv <= 3:
                m.stack_size = 300
            elif mv <= 5:
                m.stack_size = 500
            elif mv <= 10:
                m.stack_size = 500
            elif mv <= 100:
                m.stack_size = 800
            else:
                m.stack_size = 1000

    elif room == "Poker Stars":
        if reservation in ("IT", "COM"):
            for m in multipliers:
                mv = m.multiplier_value
                if mv <= 3:
                    m.stack_size = 300
                elif mv <= 5:
                    m.stack_size = 400
                else:
                    m.stack_size = 500
        else:
            if tournament_type == "flash":
                for m in multipliers:
                    m.stack_size = 300
            else:
                for m in multipliers:
                    m.stack_size = 500

    elif room == "iPoker":
        if reservation in ("ES", "FR") and tournament_type == "flash":
            for m in multipliers:
                m.stack_size = 200
        elif tournament_type == "flash":
            for m in multipliers:
                m.stack_size = 200
        else:
            pass  # regular keeps original (500)

    elif room in ("Winamax (old)", "Winamax (new)"):
        if tournament_type == "nitro":
            for m in multipliers:
                m.stack_size = 300
        else:
            for m in multipliers:
                m.stack_size = 500


def load_setups(path: Path | None = None) -> list[Setup]:
    path = path or DEFAULT_DATA_DIR / "setups.json"
    raw: list[dict] = _load_json(path)

    setups: list[Setup] = []
    for s in raw:
        multipliers = [
            Multiplier(
                item_index=m["item_index"],
                multiplier_value=m["multiplier_value"],
                approx_multiplier=m["approx_multiplier"],
                w1=m["w1"],
                w2=m["w2"],
                w3=m["w3"],
                w23=m["w23"],
                probability=m["probability"],
                frequency=m["frequency"],
                default_participation=m["default_participation"],
                max_participation=m["max_participation"],
                min_participation=m["min_participation"],
                stack_size=m["stack_size"],
            )
            for m in s["multipliers"]
        ]
        max_mult = max(m.multiplier_value for m in multipliers)
        if max_mult > 1000:
            for m in multipliers:
                if m.multiplier_value > 1000:
                    m.default_participation = 0.0

        room = s["poker_room"]
        reservation = str(s.get("reservation", ""))
        ttype = s["tournament_type"]

        # Rename legacy Winamax flash -> nitro.
        if room in ("Winamax", "Winamax (old)") and ttype == "flash":
            ttype = "nitro"

        _fix_stack_sizes(multipliers, room, reservation, ttype)

        unique_stacks = sorted(set(m.stack_size for m in multipliers))
        has_var = len(unique_stacks) > 1

        setups.append(
            Setup(
                tourney_id=s["tourney_id"],
                name=s["name"],
                poker_room=room,
                reservation=reservation,
                buyin=s["buyin"],
                tournament_type=ttype,
                default_stack=s["default_stack"],
                has_variable_stacks=has_var,
                unique_stacks=unique_stacks,
                structure_id=s["structure_id"],
                multipliers=multipliers,
            )
        )
    return setups


def _parse_distribution(d: dict) -> DiscreteDistribution:
    return DiscreteDistribution(
        values=np.array(d["values"], dtype=np.float64),
        probabilities=np.array(d["probabilities"], dtype=np.float64),
    )


def _parse_conditional_bins(raw: dict) -> dict[int, ConditionalBin]:
    bins: dict[int, ConditionalBin] = {}
    for key_str, b in raw.items():
        key = int(key_str)
        bins[key] = ConditionalBin(
            ev_range=tuple(b["ev_range"]),  # type: ignore[arg-type]
            distribution=DiscreteDistribution(
                values=np.array(b["values"], dtype=np.float64),
                probabilities=np.array(b["probabilities"], dtype=np.float64),
            ),
            n_observations=b["n_observations"],
            std=b["std"],
            original_mean=b["original_mean"],
        )
    return bins


def load_distributions(path: Path | None = None) -> dict[int, StackDistributions]:
    path = path or DEFAULT_DATA_DIR / "distributions.json"
    raw: dict[str, dict] = _load_json(path)

    result: dict[int, StackDistributions] = {}
    for stack_str, data in raw.items():
        stack = int(stack_str)

        allin_uncond = _parse_distribution(data["allin_distribution_unconditional"])
        # Center unconditional epsilon distribution to zero mean.
        # Conditional bins are already centered during preprocessing,
        # but the unconditional distribution may have a residual bias.
        uncond_mean = allin_uncond.mean
        if abs(uncond_mean) > 1e-9:
            allin_uncond = allin_uncond.shifted(-uncond_mean)

        result[stack] = StackDistributions(
            stack_size=data["stack_size"],
            n_total=data["n_total"],
            chip_ev=_parse_distribution(data["chip_ev_distribution"]),
            chip_ev_stats=data["chip_ev_stats"],
            allin_unconditional=allin_uncond,
            allin_stats_unconditional=data["allin_stats_unconditional"],
            conditional_bins=_parse_conditional_bins(
                data.get("allin_distributions_conditional", {})
            ),
        )
    return result


def load_mean_chip_ev(
    overall_path: Path | None = None,
    by_stack_path: Path | None = None,
) -> MeanChipEVStore:
    overall_path = overall_path or DEFAULT_DATA_DIR / "mean_chip_ev_overall.json"
    by_stack_path = by_stack_path or DEFAULT_DATA_DIR / "mean_chip_ev_by_stack.json"

    raw_overall: dict[str, dict] = _load_json(overall_path)
    raw_by_stack: dict[str, dict] = _load_json(by_stack_path)

    overall: dict[int, MeanChipEVEntry] = {}
    for key, v in raw_overall.items():
        overall[int(key)] = MeanChipEVEntry(
            tourney_id=v["tourney_id"],
            mean_chip_ev=v["mean_chip_ev"],
            n_observations=v["n_observations"],
        )

    by_stack: dict[str, MeanChipEVEntry] = {}
    for key, v in raw_by_stack.items():
        by_stack[key] = MeanChipEVEntry(
            tourney_id=v["tourney_id"],
            mean_chip_ev=v["mean_chip_ev"],
            n_observations=v["n_observations"],
            stack_size=v.get("stack_size"),
            median_chip_ev=v.get("median_chip_ev"),
            std_chip_ev=v.get("std_chip_ev"),
            has_variable_stacks=v.get("has_variable_stacks", False),
        )

    return MeanChipEVStore(overall=overall, by_stack=by_stack)


def load_all(
    data_dir: Path | None = None,
) -> tuple[list[Setup], dict[int, StackDistributions], MeanChipEVStore]:
    """Convenience loader: returns (setups, distributions, mean_chip_ev_store)."""
    data_dir = data_dir or DEFAULT_DATA_DIR
    setups = load_setups(data_dir / "setups.json")
    distributions = load_distributions(data_dir / "distributions.json")
    mean_ev = load_mean_chip_ev(
        data_dir / "mean_chip_ev_overall.json",
        data_dir / "mean_chip_ev_by_stack.json",
    )
    return setups, distributions, mean_ev

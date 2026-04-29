"""Import pool configuration from Excel files and manage saved profiles.

Reads an Excel file with columns:
    poker_room, reservation, buyin_total_value, starting_stack_size,
    tournaments, pool_formula, participation_mult_01..10

Maps numeric codes to domain names, groups rows into portfolio entries,
and supports saving/loading named profiles as JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import openpyxl

from pool_simulator.core.models import Setup
from pool_simulator.core.ev_formulas import EVFormula
from pool_simulator.core.paths import get_profiles_dir

PROFILES_DIR = get_profiles_dir()

ROOM_MAP: dict[int, str] = {
    1: "Poker Stars",
    2: "Winamax (new)",
    6: "Winamax (old)",
    7: "GGpoker",
    8: "iPoker",
    9: "Betclic",
}

RESERVATION_MAP: dict[int, str] = {
    0: "COM",
    1: "FR",
    2: "IT",
    3: "ES",
    5: "NJ",
    7: "BG",
    13: "PT",
    18: "PA",
}

FORMULA_MAP: dict[int, EVFormula] = {
    0: EVFormula.MULT_EV,
    1: EVFormula.AVG_EV,
    2: EVFormula.MULT_EV_50,
    3: EVFormula.MULT_EV_4,
    4: EVFormula.AW_EV,
    5: EVFormula.AVG_EV_4,
}


@dataclass
class ImportedEntry:
    """One aggregated group ready to become a PortfolioEntry."""

    poker_room: str
    reservation: str
    buyin: float
    tournament_type: str
    ev_formula: EVFormula
    n_tourneys: int
    participations: list[float | None]  # None = use default
    is_default_participation: bool


@dataclass
class PoolProfile:
    """Named collection of ImportedEntry items, serializable to JSON."""

    name: str
    entries: list[ImportedEntry]


def _determine_tournament_type(
    poker_room: str,
    reservation: str,
    stack_size: float,
) -> str:
    """Derive tournament type from stack size and room rules.

    stack=0   → variable stacks (IT/COM for PS, GGpoker, etc.)
    stack=300 → flash / nitro
    stack=200 → flash (iPoker ES/FR)
    stack=500 → regular
    """
    stack = int(round(stack_size))
    if stack == 0:
        if poker_room == "Poker Stars" and reservation in ("IT", "COM"):
            return "regular"
        if poker_room == "GGpoker":
            return "regular"
        return "regular"

    if poker_room in ("Winamax (old)", "Winamax (new)"):
        return "nitro" if stack <= 300 else "regular"
    if poker_room == "iPoker":
        return "flash" if stack <= 300 else "regular"
    if poker_room == "Poker Stars":
        if reservation in ("IT", "COM"):
            return "regular"
        return "flash" if stack <= 300 else "regular"
    return "regular"


def _parse_participation(val: Any) -> float | None:
    """Parse a participation cell; return None for 'default'."""
    if val is None:
        return None
    s = str(val).strip()
    if s.lower() in ("default", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _participation_key(parts: list[float | None]) -> tuple[float | None, ...]:
    return tuple(parts)


def import_pool_from_excel(filepath: str | Path) -> list[ImportedEntry]:
    """Read an Excel file and return grouped ImportedEntry list.

    Grouping key: (room, reservation, buyin, tournament_type, formula,
                   participation_profile).
    Tournaments are summed within each group.
    """
    wb = openpyxl.load_workbook(str(filepath), read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if not rows:
        return []

    header = rows[0]
    data_rows = rows[1:]

    groups: dict[tuple, dict[str, Any]] = {}

    for row in data_rows:
        if len(row) < 16:
            continue

        raw_room = row[0]
        raw_reservation = row[1]
        raw_buyin = row[2]
        raw_stack = row[3]
        raw_tourneys = row[4]
        raw_formula = row[5]
        raw_parts = row[6:16]

        tourneys = int(raw_tourneys) if raw_tourneys else 0
        if tourneys <= 0:
            continue

        room_code = int(raw_room)
        poker_room = ROOM_MAP.get(room_code)
        if poker_room is None:
            continue

        res_code = int(raw_reservation)
        reservation = RESERVATION_MAP.get(res_code, str(res_code))

        buyin = float(raw_buyin)
        stack_size = float(raw_stack) if raw_stack else 0.0

        formula_str = str(raw_formula).strip() if raw_formula else "NULL"
        if formula_str == "NULL":
            ev_formula = EVFormula.AVG_EV
        else:
            formula_code = int(formula_str)
            ev_formula = FORMULA_MAP.get(formula_code, EVFormula.AVG_EV)

        ttype = _determine_tournament_type(poker_room, reservation, stack_size)

        participations = [_parse_participation(p) for p in raw_parts]
        is_default = all(p is None for p in participations)
        part_key = _participation_key(participations)

        group_key = (poker_room, reservation, buyin, ttype, ev_formula, part_key)

        if group_key in groups:
            groups[group_key]["n_tourneys"] += tourneys
        else:
            groups[group_key] = {
                "poker_room": poker_room,
                "reservation": reservation,
                "buyin": buyin,
                "tournament_type": ttype,
                "ev_formula": ev_formula,
                "n_tourneys": tourneys,
                "participations": participations,
                "is_default_participation": is_default,
            }

    return [
        ImportedEntry(**g)
        for g in sorted(groups.values(), key=lambda g: (g["poker_room"], g["reservation"], g["buyin"], g["n_tourneys"]))
    ]


def match_setup(
    entry: ImportedEntry,
    setups: list[Setup],
) -> Setup | None:
    """Find the Setup that matches an ImportedEntry."""
    matches = [
        s for s in setups
        if s.poker_room == entry.poker_room
        and s.reservation == entry.reservation
        and s.tournament_type == entry.tournament_type
        and abs(s.buyin - entry.buyin) < 0.01
    ]
    return matches[0] if matches else None


def build_participations(
    entry: ImportedEntry,
    setup: Setup,
) -> list[float]:
    """Build the participation list for a setup from an ImportedEntry.

    Both participation columns and Setup.multipliers are ordered from
    highest multiplier to lowest (item_index ascending = highest mult first).
    """
    n_mult = len(setup.multipliers)

    if entry.is_default_participation:
        return [m.default_participation for m in setup.multipliers]

    imported = entry.participations

    result: list[float] = []
    for i, m in enumerate(setup.multipliers):
        if i < len(imported) and imported[i] is not None:
            result.append(imported[i])
        else:
            result.append(m.default_participation)
    return result


# ---------------------------------------------------------------------------
# Profile persistence
# ---------------------------------------------------------------------------

def _entry_to_dict(entry: ImportedEntry) -> dict:
    return {
        "poker_room": entry.poker_room,
        "reservation": entry.reservation,
        "buyin": entry.buyin,
        "tournament_type": entry.tournament_type,
        "ev_formula": entry.ev_formula.value,
        "n_tourneys": entry.n_tourneys,
        "participations": entry.participations,
        "is_default_participation": entry.is_default_participation,
    }


def _entry_from_dict(d: dict) -> ImportedEntry:
    poker_room = d["poker_room"]
    # Backward compatibility for previously saved profiles.
    if poker_room == "Winamax":
        poker_room = "Winamax (old)"
    return ImportedEntry(
        poker_room=poker_room,
        reservation=d["reservation"],
        buyin=d["buyin"],
        tournament_type=d["tournament_type"],
        ev_formula=EVFormula(d["ev_formula"]),
        n_tourneys=d["n_tourneys"],
        participations=d["participations"],
        is_default_participation=d["is_default_participation"],
    )


def save_profile(profile: PoolProfile) -> Path:
    """Save a named profile to the profiles directory. Returns the file path."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in profile.name)
    path = PROFILES_DIR / f"{safe_name}.json"

    data = {
        "name": profile.name,
        "entries": [_entry_to_dict(e) for e in profile.entries],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_profile(path: Path) -> PoolProfile:
    """Load a profile from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return PoolProfile(
        name=data["name"],
        entries=[_entry_from_dict(e) for e in data["entries"]],
    )


def list_profiles() -> list[tuple[str, Path]]:
    """Return list of (profile_name, path) for all saved profiles."""
    if not PROFILES_DIR.exists():
        return []
    result = []
    for p in sorted(PROFILES_DIR.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            result.append((data.get("name", p.stem), p))
        except (json.JSONDecodeError, KeyError):
            continue
    return result

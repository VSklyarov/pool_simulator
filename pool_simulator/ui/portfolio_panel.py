"""Portfolio (Pool mode) panel for combining multiple setup configurations."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QComboBox,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import numpy as np

from pool_simulator.core.models import (
    MeanChipEVStore,
    PortfolioEntry,
    PortfolioEntryResult,
    PortfolioResult,
    Setup,
    SimulationResult,
    StackDistributions,
    is_eur_room,
)
from pool_simulator.core.ev_formulas import EVFormula
from pool_simulator.core.simulator import compute_analytical_sigma, simulate_pool
from pool_simulator.core.pool_importer import (
    ImportedEntry,
    PoolProfile,
    build_participations,
    import_pool_from_excel,
    list_profiles,
    load_profile,
    match_setup,
    save_profile,
)


class PortfolioWorker(QThread):
    """Background thread that simulates all portfolio entries and combines results."""

    progress = Signal(int, int)
    finished = Signal(object)  # PortfolioResult
    error = Signal(str)

    def __init__(
        self,
        entries: list[PortfolioEntry],
        distributions: dict[int, StackDistributions],
        n_simulations: int,
        mode: str = "precise",
        target: str = "swing",
    ) -> None:
        super().__init__()
        self._entries = entries
        self._distributions = distributions
        self._n_simulations = n_simulations
        self._mode = mode
        self._target = target

    def run(self) -> None:
        try:
            total_steps = len(self._entries)
            combined_swings_usd = np.zeros(self._n_simulations, dtype=np.float64)
            entry_results: list[PortfolioEntryResult] = []
            total_n_tourneys = 0
            rng = np.random.default_rng()

            analytical_var_sum_usd = 0.0

            for i, entry in enumerate(self._entries):
                formula = EVFormula(entry.ev_formula_name)
                cur = entry.eur_rate if is_eur_room(
                    entry.setup.poker_room, entry.setup.reservation
                ) else 1.0

                result = simulate_pool(
                    entry.setup,
                    entry.participations,
                    entry.chip_evs,
                    self._distributions,
                    entry.n_tourneys,
                    self._n_simulations,
                    rng=rng,
                    ev_formula=formula,
                    mode=self._mode,
                    target=self._target,
                    mean_chip_ev_per_multiplier=(
                        entry.chip_evs_per_multiplier
                        if self._target != "swing" and entry.chip_evs_per_multiplier
                        else None
                    ),
                )

                swings_usd = result.swings * cur
                combined_swings_usd += swings_usd

                buyin_usd = entry.setup.buyin * cur
                sigma_sim_port_usd = float(np.std(swings_usd))
                sigma_sim_per_usd = sigma_sim_port_usd / np.sqrt(entry.n_tourneys) if entry.n_tourneys > 0 else 0.0

                sigma_a_usd = result.sigma_analytical * cur
                sigma_a_port_usd = sigma_a_usd * np.sqrt(entry.n_tourneys)
                analytical_var_sum_usd += sigma_a_port_usd ** 2

                entry_results.append(PortfolioEntryResult(
                    name=entry.name,
                    n_tourneys=entry.n_tourneys,
                    buyin_usd=buyin_usd,
                    sigma_per_tourney_usd=sigma_sim_per_usd,
                    sigma_portfolio_usd=sigma_sim_port_usd,
                    sigma_analytical_usd=sigma_a_usd,
                    sigma_analytical_portfolio_usd=sigma_a_port_usd,
                    swings_usd=swings_usd,
                    poker_room=entry.setup.poker_room,
                    reservation=entry.setup.reservation,
                    ev_formula_name=entry.ev_formula_name,
                    buyin_native=entry.setup.buyin,
                ))
                total_n_tourneys += entry.n_tourneys
                self.progress.emit(i + 1, total_steps)

            combined_std = float(np.std(combined_swings_usd))
            combined_per_tourney = combined_std / np.sqrt(total_n_tourneys) if total_n_tourneys > 0 else 0.0

            analytical_sigma_portfolio = float(np.sqrt(analytical_var_sum_usd))
            analytical_sigma_per = analytical_sigma_portfolio / np.sqrt(total_n_tourneys) if total_n_tourneys > 0 else 0.0

            pct_keys = [1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0]
            pct_vals = np.percentile(combined_swings_usd, pct_keys)
            percentiles = dict(zip(pct_keys, pct_vals.tolist()))

            ci_95 = (float(np.percentile(combined_swings_usd, 2.5)),
                     float(np.percentile(combined_swings_usd, 97.5)))
            ci_99 = (float(np.percentile(combined_swings_usd, 0.5)),
                     float(np.percentile(combined_swings_usd, 99.5)))

            combined_result = SimulationResult(
                swings=combined_swings_usd,
                metric_label=result.metric_label if entry_results else "Swing",
                sigma_analytical=analytical_sigma_per,
                sigma_analytical_portfolio=analytical_sigma_portfolio,
                sigma_simulated=combined_per_tourney,
                sigma_simulated_portfolio=combined_std,
                ci_95=ci_95,
                ci_99=ci_99,
                percentiles=percentiles,
                prob_positive=float(np.mean(combined_swings_usd > 0)),
                prob_negative=float(np.mean(combined_swings_usd < 0)),
                mean_swing=float(np.mean(combined_swings_usd)),
                n_tourneys=total_n_tourneys,
                n_simulations=self._n_simulations,
                config_name="Portfolio",
                buyin=1.0,
            )

            portfolio_result = PortfolioResult(
                entry_results=entry_results,
                combined=combined_result,
                total_n_tourneys=total_n_tourneys,
            )
            self.finished.emit(portfolio_result)
        except Exception as e:
            self.error.emit(str(e))


SIMULATION_PRESETS = [
    ("100K", 100_000),
    ("500K", 500_000),
    ("1M", 1_000_000),
    ("5M", 5_000_000),
    ("10M", 10_000_000),
    ("50M", 50_000_000),
    ("100M", 100_000_000),
]


class PortfolioPanel(QGroupBox):
    """UI panel for building and simulating a portfolio of setups."""

    add_clicked = Signal()
    update_clicked = Signal()
    simulation_finished = Signal(object)  # PortfolioResult
    entry_selected = Signal(object)  # PortfolioEntry (or None)

    TABLE_COLS = ["Setup", "N Tourneys", "EV Formula", "Buy-in ($)"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Pool Mode (Portfolio)", parent)
        self._entries: list[PortfolioEntry] = []
        self._distributions: dict[int, StackDistributions] = {}
        self._setups: list[Setup] = []
        self._mean_ev_store: MeanChipEVStore | None = None
        self._eur_rate: float = 1.16
        self._worker: PortfolioWorker | None = None
        self._target_metric: str = "swing"

        layout = QVBoxLayout(self)

        self._table = QTableWidget(0, len(self.TABLE_COLS))
        self._table.setHorizontalHeaderLabels(self.TABLE_COLS)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        for i in range(len(self.TABLE_COLS)):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
        self._table.setMaximumHeight(150)
        self._table.clicked.connect(self._on_entry_clicked)
        layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("Add to Portfolio")
        self._btn_add.clicked.connect(self.add_clicked.emit)
        btn_row.addWidget(self._btn_add)

        self._btn_update = QPushButton("Update Selected")
        self._btn_update.clicked.connect(self._on_update_entry)
        btn_row.addWidget(self._btn_update)

        self._btn_remove = QPushButton("Remove")
        self._btn_remove.clicked.connect(self._on_remove)
        btn_row.addWidget(self._btn_remove)

        self._btn_clear = QPushButton("Clear")
        self._btn_clear.clicked.connect(self._on_clear)
        btn_row.addWidget(self._btn_clear)

        layout.addLayout(btn_row)

        # Import / Save / Load row
        io_row = QHBoxLayout()

        self._btn_import = QPushButton("Import from Excel")
        self._btn_import.clicked.connect(self._on_import)
        io_row.addWidget(self._btn_import)

        self._btn_save = QPushButton("Save Profile")
        self._btn_save.clicked.connect(self._on_save_profile)
        io_row.addWidget(self._btn_save)

        self._btn_load = QPushButton("Load Profile")
        self._btn_load.clicked.connect(self._on_load_profile)
        io_row.addWidget(self._btn_load)

        layout.addLayout(io_row)

        # EUR rate for portfolio
        rate_row = QHBoxLayout()
        rate_row.addWidget(QLabel("EUR/USD:"))
        self._spin_eur_rate = QDoubleSpinBox()
        self._spin_eur_rate.setRange(0.50, 3.00)
        self._spin_eur_rate.setSingleStep(0.01)
        self._spin_eur_rate.setDecimals(4)
        self._spin_eur_rate.setValue(1.1600)
        self._spin_eur_rate.valueChanged.connect(self._on_eur_rate_changed)
        rate_row.addWidget(self._spin_eur_rate)
        rate_row.addStretch()
        layout.addLayout(rate_row)

        # Simulation controls
        sim_row = QHBoxLayout()
        self._combo_sims = QComboBox()
        for label, _ in SIMULATION_PRESETS:
            self._combo_sims.addItem(label)
        self._combo_sims.setCurrentIndex(2)
        sim_row.addWidget(QLabel("Sims:"))
        sim_row.addWidget(self._combo_sims)

        self._combo_mode = QComboBox()
        self._combo_mode.addItem("Eco", "eco")
        self._combo_mode.addItem("No ChipEV Var", "no_chip_ev_var")
        self._combo_mode.addItem("Precise", "precise")
        self._combo_mode.setCurrentIndex(0)
        sim_row.addWidget(QLabel("Mode:"))
        sim_row.addWidget(self._combo_mode)

        layout.addLayout(sim_row)

        self._btn_run = QPushButton("Run Portfolio Simulation")
        self._btn_run.clicked.connect(self._on_run)
        layout.addWidget(self._btn_run)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

    def set_distributions(self, distributions: dict[int, StackDistributions]) -> None:
        self._distributions = distributions

    def set_setups(self, setups: list[Setup]) -> None:
        self._setups = setups

    def set_mean_ev_store(self, store: MeanChipEVStore) -> None:
        self._mean_ev_store = store

    def set_eur_rate(self, rate: float) -> None:
        self._eur_rate = rate
        self._spin_eur_rate.blockSignals(True)
        self._spin_eur_rate.setValue(rate)
        self._spin_eur_rate.blockSignals(False)

    def set_eur_rate_fn(self, fn) -> None:
        """Set a callable that returns the current EUR/USD rate."""
        self._eur_rate_fn = fn

    def _current_eur_rate(self) -> float:
        return self._spin_eur_rate.value()

    def _on_eur_rate_changed(self, value: float) -> None:
        self._eur_rate = value
        for entry in self._entries:
            entry.eur_rate = value
        self._refresh_table()

    def add_entry(
        self,
        setup: Setup,
        participations: list[float],
        chip_evs: dict[int, float],
        ev_formula: EVFormula,
        n_tourneys: int,
        eur_rate: float = 1.0,
        chip_evs_per_multiplier: list[float] | None = None,
    ) -> None:
        cur = eur_rate if is_eur_room(setup.poker_room, setup.reservation) else 1.0
        entry = PortfolioEntry(
            name=setup.name,
            setup=setup,
            participations=list(participations),
            chip_evs=dict(chip_evs),
            chip_evs_per_multiplier=list(chip_evs_per_multiplier or []),
            ev_formula_name=ev_formula.value,
            n_tourneys=n_tourneys,
            eur_rate=eur_rate,
        )
        self._entries.append(entry)
        self._refresh_table()

    def _refresh_table(self) -> None:
        self._table.setRowCount(len(self._entries))
        for row, entry in enumerate(self._entries):
            cur = entry.eur_rate if is_eur_room(
                entry.setup.poker_room, entry.setup.reservation
            ) else 1.0
            buyin_usd = entry.setup.buyin * cur
            self._table.setItem(row, 0, self._ro(entry.name))
            self._table.setItem(row, 1, self._ro(f"{entry.n_tourneys:,}"))
            self._table.setItem(row, 2, self._ro(entry.ev_formula_name))
            self._table.setItem(row, 3, self._ro(f"${buyin_usd:g}"))

    def _on_entry_clicked(self, index) -> None:
        row = index.row()
        if 0 <= row < len(self._entries):
            self.entry_selected.emit(self._entries[row])

    def _on_update_entry(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            QMessageBox.information(self, "Update", "Select an entry first.")
            return
        self.update_clicked.emit()

    def update_selected_entry(
        self,
        participations: list[float],
        chip_evs: dict[int, float],
        chip_evs_per_multiplier: list[float] | None,
        ev_formula: EVFormula,
        n_tourneys: int,
    ) -> None:
        """Update the currently selected entry with new parameters."""
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if 0 <= idx < len(self._entries):
            entry = self._entries[idx]
            self._entries[idx] = PortfolioEntry(
                name=entry.name,
                setup=entry.setup,
                participations=list(participations),
                chip_evs=dict(chip_evs),
                chip_evs_per_multiplier=list(chip_evs_per_multiplier or []),
                ev_formula_name=ev_formula.value,
                n_tourneys=n_tourneys,
                eur_rate=entry.eur_rate,
            )
            self._refresh_table()

    def _on_remove(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if 0 <= idx < len(self._entries):
            self._entries.pop(idx)
            self._refresh_table()

    def _on_run(self) -> None:
        if not self._entries:
            return
        if self._worker is not None and self._worker.isRunning():
            return

        n_sims = SIMULATION_PRESETS[self._combo_sims.currentIndex()][1]

        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)

        mode = self._combo_mode.currentData()
        self._worker = PortfolioWorker(
            self._entries,
            self._distributions,
            n_sims,
            mode=mode,
            target=self._target_metric,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def set_target_metric(self, target: str) -> None:
        self._target_metric = target or "swing"

    def _on_progress(self, current: int, total: int) -> None:
        pct = int(100 * current / total) if total else 0
        self._progress.setValue(pct)

    def _on_finished(self, result) -> None:
        self._btn_run.setEnabled(True)
        self._progress.setVisible(False)
        self.simulation_finished.emit(result)

    def _on_error(self, message: str) -> None:
        self._btn_run.setEnabled(True)
        self._progress.setVisible(False)
        QMessageBox.critical(self, "Portfolio Simulation Error", message)

    def _on_clear(self) -> None:
        self._entries.clear()
        self._refresh_table()

    # ------------------------------------------------------------------
    # Import from Excel
    # ------------------------------------------------------------------

    def _on_import(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Import Pool from Excel",
            "",
            "Excel Files (*.xlsx *.xls);;All Files (*)",
        )
        if not filepath:
            return

        try:
            imported = import_pool_from_excel(filepath)
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", str(exc))
            return

        if not imported:
            QMessageBox.information(self, "Import", "No entries found in the file.")
            return

        eur_rate = self._current_eur_rate()
        added = 0
        skipped: list[str] = []
        for ie in imported:
            setup = match_setup(ie, self._setups)
            if setup is None:
                skipped.append(
                    f"{ie.poker_room} {ie.reservation} "
                    f"buyin={ie.buyin} type={ie.tournament_type}"
                )
                continue

            parts = build_participations(ie, setup)
            chip_evs = self._get_default_chip_evs(setup)
            chip_evs_per_multiplier = self._build_chip_evs_per_multiplier(setup, chip_evs)

            self.add_entry(
                setup=setup,
                participations=parts,
                chip_evs=chip_evs,
                chip_evs_per_multiplier=chip_evs_per_multiplier,
                ev_formula=ie.ev_formula,
                n_tourneys=ie.n_tourneys,
                eur_rate=eur_rate,
            )
            added += 1

        msg = f"Imported {added} entries ({sum(e.n_tourneys for e in self._entries[-added:]):,} tournaments)."
        if skipped:
            msg += f"\n\nSkipped {len(skipped)} entries (no matching setup):\n"
            msg += "\n".join(skipped[:10])
            if len(skipped) > 10:
                msg += f"\n...and {len(skipped) - 10} more"

        QMessageBox.information(self, "Import Complete", msg)

    def _get_default_chip_evs(self, setup: Setup) -> dict[int, float]:
        if self._mean_ev_store is None:
            return {}
        defaults = self._mean_ev_store.get_default(setup)
        if not defaults:
            return {}
        if setup.has_variable_stacks:
            min_stack = min(setup.unique_stacks) if setup.unique_stacks else setup.default_stack
            min_cev = defaults.get(min_stack, 0.0)
            for stack in setup.unique_stacks:
                if stack not in defaults or defaults[stack] == 0.0:
                    defaults[stack] = (
                        min_cev * stack / min_stack if min_stack > 0 else 0.0
                    )
        return defaults

    # ------------------------------------------------------------------
    # Save / Load profiles
    # ------------------------------------------------------------------

    def _on_save_profile(self) -> None:
        if not self._entries:
            QMessageBox.warning(self, "Save Profile", "Portfolio is empty.")
            return

        name, ok = QInputDialog.getText(
            self, "Save Profile", "Profile name:"
        )
        if not ok or not name.strip():
            return

        imported_entries = self._entries_to_imported()
        profile = PoolProfile(name=name.strip(), entries=imported_entries)

        try:
            path = save_profile(profile)
            QMessageBox.information(
                self, "Profile Saved",
                f"Saved as: {path.name}\n{len(imported_entries)} entries."
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _entries_to_imported(self) -> list[ImportedEntry]:
        """Convert current PortfolioEntry list back to ImportedEntry for saving."""
        result: list[ImportedEntry] = []
        for pe in self._entries:
            defaults = [m.default_participation for m in pe.setup.multipliers]
            is_default = pe.participations == defaults
            parts: list[float | None] = (
                [None] * len(pe.participations)
                if is_default
                else list(pe.participations)
            )
            result.append(ImportedEntry(
                poker_room=pe.setup.poker_room,
                reservation=pe.setup.reservation,
                buyin=pe.setup.buyin,
                tournament_type=pe.setup.tournament_type,
                ev_formula=EVFormula(pe.ev_formula_name),
                n_tourneys=pe.n_tourneys,
                participations=parts,
                is_default_participation=is_default,
            ))
        return result

    def _on_load_profile(self) -> None:
        profiles = list_profiles()
        if not profiles:
            QMessageBox.information(
                self, "Load Profile",
                "No saved profiles found."
            )
            return

        names = [name for name, _ in profiles]
        name, ok = QInputDialog.getItem(
            self, "Load Profile", "Select profile:", names, 0, False
        )
        if not ok:
            return

        idx = names.index(name)
        _, path = profiles[idx]

        try:
            profile = load_profile(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        self._entries.clear()
        eur_rate = self._current_eur_rate()
        skipped: list[str] = []
        for ie in profile.entries:
            setup = match_setup(ie, self._setups)
            if setup is None:
                skipped.append(
                    f"{ie.poker_room} {ie.reservation} "
                    f"buyin={ie.buyin} type={ie.tournament_type}"
                )
                continue
            parts = build_participations(ie, setup)
            chip_evs = self._get_default_chip_evs(setup)
            chip_evs_per_multiplier = self._build_chip_evs_per_multiplier(setup, chip_evs)
            self.add_entry(
                setup=setup,
                participations=parts,
                chip_evs=chip_evs,
                chip_evs_per_multiplier=chip_evs_per_multiplier,
                ev_formula=ie.ev_formula,
                n_tourneys=ie.n_tourneys,
                eur_rate=eur_rate,
            )

        msg = f"Loaded profile '{profile.name}': {len(self._entries)} entries."
        if skipped:
            msg += f"\nSkipped {len(skipped)} (no matching setup)."
        QMessageBox.information(self, "Profile Loaded", msg)

    @staticmethod
    def _ro(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        return item

    @staticmethod
    def _build_chip_evs_per_multiplier(
        setup: Setup,
        chip_evs: dict[int, float],
    ) -> list[float]:
        """Build per-multiplier ChipEV list in setup multiplier order."""
        return [float(chip_evs.get(m.stack_size, 0.0)) for m in setup.multipliers]

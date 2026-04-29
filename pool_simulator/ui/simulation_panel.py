"""Simulation parameters panel with Run button and worker thread."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pool_simulator.core.models import (
    MeanChipEVStore,
    Setup,
    SimulationResult,
    StackDistributions,
    is_eur_room,
)
from pool_simulator.core.ev_formulas import EVFormula
from pool_simulator.core.simulator import compute_analytical_sigma, simulate_pool


class SimulationWorker(QThread):
    """Background thread for running the Monte Carlo simulation."""

    progress = Signal(int, int)
    finished = Signal(object)  # SimulationResult
    error = Signal(str)

    def __init__(
        self,
        setup: Setup,
        participations: list[float],
        mean_chip_ev: dict[int, float],
        mean_chip_ev_per_multiplier: list[float] | None,
        distributions: dict[int, StackDistributions],
        n_tourneys: int,
        n_simulations: int,
        ev_formula: EVFormula = EVFormula.AVG_EV,
        mode: str = "precise",
        target: str = "swing",
    ) -> None:
        super().__init__()
        self._setup = setup
        self._participations = participations
        self._mean_chip_ev = mean_chip_ev
        self._mean_chip_ev_per_multiplier = mean_chip_ev_per_multiplier
        self._distributions = distributions
        self._n_tourneys = n_tourneys
        self._n_simulations = n_simulations
        self._ev_formula = ev_formula
        self._mode = mode
        self._target = target

    def run(self) -> None:
        try:
            result = simulate_pool(
                self._setup,
                self._participations,
                self._mean_chip_ev,
                self._distributions,
                self._n_tourneys,
                self._n_simulations,
                ev_formula=self._ev_formula,
                progress_callback=self._report_progress,
                mode=self._mode,
                target=self._target,
                mean_chip_ev_per_multiplier=self._mean_chip_ev_per_multiplier,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def _report_progress(self, current: int, total: int) -> None:
        self.progress.emit(current, total)


class SimulationPanel(QGroupBox):
    """Parameters input + Run button + progress bar."""

    simulation_finished = Signal(object)  # SimulationResult
    target_changed = Signal(str)

    SIMULATION_PRESETS = [
        ("100K", 100_000),
        ("500K", 500_000),
        ("1M", 1_000_000),
        ("5M", 5_000_000),
        ("10M", 10_000_000),
        ("50M", 50_000_000),
        ("100M", 100_000_000),
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Simulation Parameters", parent)
        self._worker: SimulationWorker | None = None
        self._setup: Setup | None = None
        self._distributions: dict[int, StackDistributions] = {}
        self._participations_fn = lambda: []
        self._chip_evs_fn = lambda: {}
        self._mean_ev_store: MeanChipEVStore | None = None

        form = QFormLayout()

        self._combo_target = QComboBox()
        self._combo_target.addItem("Swing (Result − EV)", "swing")
        self._combo_target.addItem("EV (by chosen formula)", "ev")
        self._combo_target.setCurrentIndex(0)
        form.addRow("Target:", self._combo_target)

        self._spin_tourneys = QSpinBox()
        self._spin_tourneys.setRange(1_000, 100_000_000)
        self._spin_tourneys.setSingleStep(10_000)
        self._spin_tourneys.setValue(100_000)
        self._spin_tourneys.setGroupSeparatorShown(True)
        form.addRow("Tournaments (N):", self._spin_tourneys)

        self._spin_chip_ev = QDoubleSpinBox()
        self._spin_chip_ev.setRange(-1000.0, 1000.0)
        self._spin_chip_ev.setSingleStep(1.0)
        self._spin_chip_ev.setDecimals(2)
        self._spin_chip_ev.setValue(0.0)
        form.addRow("Mean ChipEV:", self._spin_chip_ev)

        self._combo_ev_formula = QComboBox()
        for f in EVFormula:
            self._combo_ev_formula.addItem(f.value, f)
        self._combo_ev_formula.setCurrentIndex(0)
        form.addRow("EV Formula:", self._combo_ev_formula)

        self._spin_eur_rate = QDoubleSpinBox()
        self._spin_eur_rate.setRange(0.50, 3.00)
        self._spin_eur_rate.setSingleStep(0.01)
        self._spin_eur_rate.setDecimals(4)
        self._spin_eur_rate.setValue(1.1600)
        form.addRow("EUR/USD rate:", self._spin_eur_rate)

        self._combo_sims = QComboBox()
        for label, _ in self.SIMULATION_PRESETS:
            self._combo_sims.addItem(label)
        self._combo_sims.setCurrentIndex(2)  # default 1M
        form.addRow("Simulations (M):", self._combo_sims)

        self._combo_mode = QComboBox()
        self._combo_mode.addItem("Eco (fast, ~3-5% σ error)", "eco")
        self._combo_mode.addItem("No ChipEV Var (fast, fixed ChipEV)", "no_chip_ev_var")
        self._combo_mode.addItem("Precise (exact, <2% σ error)", "precise")
        self._combo_mode.setCurrentIndex(0)
        form.addRow("Mode:", self._combo_mode)

        self._label_sigma = QLabel("—")
        form.addRow("Analytical σ (per tourney):", self._label_sigma)

        self._label_sigma_portfolio = QLabel("—")
        form.addRow("Analytical σ (portfolio):", self._label_sigma_portfolio)

        self._btn_run = QPushButton("Run Simulation")
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._on_run_clicked)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(form)
        main_layout.addWidget(self._btn_run)
        main_layout.addWidget(self._progress)

        self._spin_tourneys.valueChanged.connect(self._update_analytical)
        self._spin_chip_ev.valueChanged.connect(self._on_global_chip_ev_changed)
        self._combo_ev_formula.currentIndexChanged.connect(self._update_analytical)
        self._combo_target.currentIndexChanged.connect(self._update_analytical)
        self._combo_target.currentIndexChanged.connect(
            lambda: self.target_changed.emit(self.target_metric)
        )

        self._global_chip_ev_callback = None

    @property
    def ev_formula(self) -> EVFormula:
        return self._combo_ev_formula.currentData()

    @property
    def target_metric(self) -> str:
        return self._combo_target.currentData()

    @property
    def eur_rate(self) -> float:
        return self._spin_eur_rate.value()

    @property
    def simulation_mode(self) -> str:
        return self._combo_mode.currentData()

    def get_currency_multiplier(self, setup: Setup | None = None) -> float:
        """Return multiplier to convert room-native currency to USD."""
        s = setup or self._setup
        if s is None:
            return 1.0
        if is_eur_room(s.poker_room, s.reservation):
            return self._spin_eur_rate.value()
        return 1.0

    def set_participations_fn(self, fn) -> None:
        self._participations_fn = fn

    def set_chip_evs_fn(self, fn) -> None:
        self._chip_evs_fn = fn

    def set_global_chip_ev_callback(self, fn) -> None:
        """Set callback to propagate global ChipEV changes to ParticipationEditor."""
        self._global_chip_ev_callback = fn

    def set_distributions(self, distributions: dict[int, StackDistributions]) -> None:
        self._distributions = distributions

    def set_mean_ev_store(self, store: MeanChipEVStore) -> None:
        self._mean_ev_store = store

    def on_setup_changed(self, setup: Setup | None) -> None:
        self._setup = setup
        self._btn_run.setEnabled(setup is not None)

        if setup is None:
            self._spin_chip_ev.setValue(0.0)
            self._label_sigma.setText("—")
            self._label_sigma_portfolio.setText("—")
            return

        if self._mean_ev_store is not None:
            defaults = self._mean_ev_store.get_default(setup)
            if defaults:
                min_stack = min(setup.unique_stacks) if setup.unique_stacks else setup.default_stack
                global_val = defaults.get(min_stack, 0.0)
                if global_val == 0.0 and defaults:
                    global_val = next(iter(defaults.values()))
                self._spin_chip_ev.blockSignals(True)
                self._spin_chip_ev.setValue(global_val)
                self._spin_chip_ev.blockSignals(False)

        self._update_analytical()

    def _on_global_chip_ev_changed(self) -> None:
        if self._global_chip_ev_callback is not None:
            self._global_chip_ev_callback(self._spin_chip_ev.value())
        self._update_analytical()

    def _update_analytical(self) -> None:
        if self._setup is None:
            return

        participations = self._participations_fn()
        if not participations:
            return

        if self.target_metric != "swing":
            self._label_sigma.setText("—")
            self._label_sigma_portfolio.setText("—")
            return

        mean_ev = self._chip_evs_fn()
        if not mean_ev:
            mean_ev = self._get_mean_chip_ev_dict()

        formula = self.ev_formula
        try:
            sigma = compute_analytical_sigma(
                self._setup, participations, mean_ev, self._distributions,
                ev_formula=formula,
            )
            cur = self.get_currency_multiplier()
            sigma_usd = sigma * cur
            n = self._spin_tourneys.value()
            sigma_port_usd = sigma_usd * (n ** 0.5)
            buyin_usd = self._setup.buyin * cur

            self._label_sigma.setText(
                f"${sigma_usd:.4f}  ({sigma_usd / buyin_usd:.4f} BI)"
            )
            self._label_sigma_portfolio.setText(
                f"${sigma_port_usd:,.2f}  ({sigma_port_usd / buyin_usd:,.2f} BI)"
            )
        except Exception:
            self._label_sigma.setText("Error")
            self._label_sigma_portfolio.setText("Error")

    def _get_mean_chip_ev_dict(self) -> dict[int, float]:
        if self._setup is None:
            return {}
        mean_val = self._spin_chip_ev.value()
        if self._setup.has_variable_stacks:
            min_stack = min(self._setup.unique_stacks) if self._setup.unique_stacks else self._setup.default_stack
            return {s: mean_val * s / min_stack for s in self._setup.unique_stacks}
        return {self._setup.default_stack: mean_val}

    def _on_run_clicked(self) -> None:
        if self._setup is None:
            return
        if self._worker is not None and self._worker.isRunning():
            return

        participations = self._participations_fn()
        mean_ev = self._chip_evs_fn()
        if not mean_ev:
            mean_ev = self._get_mean_chip_ev_dict()
        mean_ev_per_mult = None
        if self.target_metric != "swing":
            # We need per-multiplier ChipEV even if stacks repeat (e.g., regular 500).
            owner = getattr(self._chip_evs_fn, "__self__", None)
            fn = getattr(owner, "get_chip_evs_per_multiplier", None) if owner is not None else None
            if callable(fn):
                mean_ev_per_mult = fn()
        n_sims = self.SIMULATION_PRESETS[self._combo_sims.currentIndex()][1]
        n_tourneys = self._spin_tourneys.value()

        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)

        self._worker = SimulationWorker(
            self._setup,
            participations,
            mean_ev,
            mean_ev_per_mult,
            self._distributions,
            n_tourneys,
            n_sims,
            ev_formula=self.ev_formula,
            mode=self.simulation_mode,
            target=self.target_metric,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, current: int, total: int) -> None:
        pct = int(100 * current / total) if total else 0
        self._progress.setValue(pct)

    def _on_finished(self, result: SimulationResult) -> None:
        self._btn_run.setEnabled(True)
        self._progress.setVisible(False)
        self.simulation_finished.emit(result)

    def _on_error(self, message: str) -> None:
        self._btn_run.setEnabled(True)
        self._progress.setVisible(False)
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Simulation Error", message)

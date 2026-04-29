"""Main application window — assembles all UI panels."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from pool_simulator.core.data_loader import load_all
from pool_simulator.core.models import (
    MeanChipEVStore,
    PortfolioEntry,
    PortfolioResult,
    Setup,
    SimulationResult,
    StackDistributions,
    is_eur_room,
)
from pool_simulator.core.ev_formulas import EVFormula
from pool_simulator.ui.participation_editor import ParticipationEditor
from pool_simulator.ui.results_panel import ResultsPanel
from pool_simulator.ui.setup_selector import SetupSelector
from pool_simulator.ui.simulation_panel import SimulationPanel
from pool_simulator.ui.portfolio_panel import PortfolioPanel


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pool Simulator")
        self.setMinimumSize(1100, 700)

        self._setups: list[Setup] = []
        self._distributions: dict[int, StackDistributions] = {}
        self._mean_ev_store: MeanChipEVStore | None = None

        self._build_ui()
        self._load_data()
        self._connect_signals()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)

        self._setup_selector = SetupSelector([])
        left_layout.addWidget(self._setup_selector)

        self._participation_editor = ParticipationEditor()
        left_layout.addWidget(self._participation_editor)

        self._simulation_panel = SimulationPanel()
        left_layout.addWidget(self._simulation_panel)

        self._portfolio_panel = PortfolioPanel()
        left_layout.addWidget(self._portfolio_panel)

        left_layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_widget)
        left_scroll.setMinimumWidth(420)
        left_scroll.setMaximumWidth(580)

        self._results_panel = ResultsPanel()

        splitter.addWidget(left_scroll)
        splitter.addWidget(self._results_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.addWidget(splitter)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

    def _load_data(self) -> None:
        self._status_bar.showMessage("Loading data...")
        try:
            self._setups, self._distributions, self._mean_ev_store = load_all()
            self._setup_selector = SetupSelector(self._setups)

            left_widget = self.centralWidget().findChild(QScrollArea).widget()
            old_selector = left_widget.layout().itemAt(0).widget()
            left_widget.layout().replaceWidget(old_selector, self._setup_selector)
            old_selector.deleteLater()

            self._simulation_panel.set_distributions(self._distributions)
            self._simulation_panel.set_mean_ev_store(self._mean_ev_store)
            self._portfolio_panel.set_distributions(self._distributions)
            self._portfolio_panel.set_setups(self._setups)
            self._portfolio_panel.set_mean_ev_store(self._mean_ev_store)

            self._status_bar.showMessage(
                f"Loaded {len(self._setups)} setups, "
                f"{len(self._distributions)} stack distributions"
            )
        except Exception as e:
            self._status_bar.showMessage(f"Error loading data: {e}")

    def _connect_signals(self) -> None:
        self._setup_selector.setup_changed.connect(self._on_setup_changed)
        self._participation_editor.participation_changed.connect(
            self._on_participation_changed
        )
        self._participation_editor.chip_ev_changed.connect(
            self._on_chip_ev_changed
        )
        self._simulation_panel.set_participations_fn(
            self._participation_editor.get_participations
        )
        self._simulation_panel.set_chip_evs_fn(
            self._participation_editor.get_chip_evs_dict
        )
        self._simulation_panel.set_global_chip_ev_callback(
            self._participation_editor.set_chip_evs_from_global
        )
        self._simulation_panel.simulation_finished.connect(self._on_simulation_finished)
        self._simulation_panel.target_changed.connect(self._portfolio_panel.set_target_metric)

        self._portfolio_panel.add_clicked.connect(self._on_add_to_portfolio)
        self._portfolio_panel.update_clicked.connect(self._on_update_portfolio_entry)
        self._portfolio_panel.simulation_finished.connect(self._on_simulation_finished)
        self._portfolio_panel.entry_selected.connect(self._on_portfolio_entry_selected)
        self._portfolio_panel.set_eur_rate_fn(
            lambda: self._simulation_panel.eur_rate
        )

    def _on_setup_changed(self, setup: Setup | None) -> None:
        if setup is None:
            self._participation_editor.set_multipliers([])
            self._results_panel.clear_results()
        else:
            default_evs = self._get_default_chip_evs(setup)
            self._participation_editor.set_multipliers(
                setup.multipliers,
                default_chip_evs=default_evs,
                has_variable_stacks=setup.has_variable_stacks,
            )
            self._results_panel.set_buyin(setup.buyin)

        self._simulation_panel.on_setup_changed(setup)

    def _get_default_chip_evs(self, setup: Setup) -> dict[int, float]:
        """Build {stack_size: chip_ev} for the setup using MeanChipEVStore."""
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
                    defaults[stack] = min_cev * stack / min_stack if min_stack > 0 else 0.0
        return defaults

    def _on_participation_changed(self) -> None:
        self._simulation_panel._update_analytical()

    def _on_chip_ev_changed(self) -> None:
        self._simulation_panel._update_analytical()

    def _on_add_to_portfolio(self) -> None:
        setup = self._setup_selector.current_setup
        if setup is None:
            return
        participations = self._participation_editor.get_participations()
        chip_evs = self._participation_editor.get_chip_evs_dict()
        chip_evs_per_multiplier = self._participation_editor.get_chip_evs_per_multiplier()
        ev_formula = self._simulation_panel.ev_formula
        n_tourneys = self._simulation_panel._spin_tourneys.value()
        eur_rate = self._simulation_panel.eur_rate

        self._portfolio_panel.set_eur_rate(eur_rate)
        self._portfolio_panel.add_entry(
            setup=setup,
            participations=participations,
            chip_evs=chip_evs,
            chip_evs_per_multiplier=chip_evs_per_multiplier,
            ev_formula=ev_formula,
            n_tourneys=n_tourneys,
            eur_rate=eur_rate,
        )

    def _on_update_portfolio_entry(self) -> None:
        """Push current UI params into the selected portfolio entry."""
        participations = self._participation_editor.get_participations()
        chip_evs = self._participation_editor.get_chip_evs_dict()
        chip_evs_per_multiplier = self._participation_editor.get_chip_evs_per_multiplier()
        ev_formula = self._simulation_panel.ev_formula
        n_tourneys = self._simulation_panel._spin_tourneys.value()
        self._portfolio_panel.update_selected_entry(
            participations, chip_evs, chip_evs_per_multiplier, ev_formula, n_tourneys,
        )
        self._status_bar.showMessage("Portfolio entry updated.")

    def _on_portfolio_entry_selected(self, entry: PortfolioEntry) -> None:
        """When user clicks an entry in the portfolio table, load its params into the UI."""
        setup = entry.setup
        self._setup_selector.select_setup(setup)

        self._participation_editor.set_multipliers(
            setup.multipliers,
            default_chip_evs={s: entry.chip_evs.get(s, 0.0) for s in setup.unique_stacks},
            has_variable_stacks=setup.has_variable_stacks,
        )
        self._participation_editor.set_participations(entry.participations)
        self._participation_editor.set_chip_evs(entry.chip_evs)

        formula = EVFormula(entry.ev_formula_name)
        idx = self._simulation_panel._combo_ev_formula.findData(formula)
        if idx >= 0:
            self._simulation_panel._combo_ev_formula.setCurrentIndex(idx)
        self._simulation_panel._spin_tourneys.setValue(entry.n_tourneys)

        self._status_bar.showMessage(f"Loaded portfolio entry: {entry.name}")

    def _on_simulation_finished(self, result) -> None:
        if isinstance(result, PortfolioResult):
            self._results_panel.show_portfolio_result(result)
            self._status_bar.showMessage(
                f"Portfolio simulation complete: "
                f"{result.combined.n_simulations:,} simulations, "
                f"{result.total_n_tourneys:,} total tournaments"
            )
        elif isinstance(result, SimulationResult):
            cur = self._simulation_panel.get_currency_multiplier()
            if cur != 1.0:
                import numpy as np
                result_usd = SimulationResult(
                    swings=result.swings * cur,
                    metric_label=result.metric_label,
                    sigma_analytical=result.sigma_analytical * cur,
                    sigma_analytical_portfolio=result.sigma_analytical_portfolio * cur,
                    sigma_simulated=result.sigma_simulated * cur,
                    sigma_simulated_portfolio=result.sigma_simulated_portfolio * cur,
                    ci_95=(result.ci_95[0] * cur, result.ci_95[1] * cur),
                    ci_99=(result.ci_99[0] * cur, result.ci_99[1] * cur),
                    percentiles={k: v * cur for k, v in result.percentiles.items()},
                    prob_positive=result.prob_positive,
                    prob_negative=result.prob_negative,
                    mean_swing=result.mean_swing * cur,
                    n_tourneys=result.n_tourneys,
                    n_simulations=result.n_simulations,
                    config_name=result.config_name,
                    buyin=result.buyin * cur,
                    poker_room=result.poker_room,
                    reservation=result.reservation,
                    buyin_native=result.buyin_native,
                    ev_formula_name=result.ev_formula_name,
                )
                result = result_usd
            self._results_panel.show_result(result)
            self._status_bar.showMessage(
                f"Simulation complete: {result.n_simulations:,} simulations of "
                f"{result.n_tourneys:,} tournaments"
            )

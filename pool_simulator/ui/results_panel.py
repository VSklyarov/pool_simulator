"""Results display: histogram, percentiles table, confidence intervals, summary."""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from pathlib import Path
import csv

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pool_simulator.core.models import (
    PortfolioEntryResult,
    PortfolioResult,
    SimulationResult,
)


def _fmt_thousands(x: float, _pos=None) -> str:
    """Format number with space as thousands separator."""
    if abs(x) < 1:
        return f"{x:.4f}"
    s = f"{x:,.0f}"
    return s.replace(",", " ")


def _fmt_thousands_2(v: float) -> str:
    s = f"{v:,.2f}"
    return s.replace(",", " ")


def _fmt_thousands_int(v: float) -> str:
    s = f"{v:,.0f}"
    return s.replace(",", " ")


class HistogramCanvas(FigureCanvas):
    """Matplotlib canvas with KDE-smoothed histogram and interactive tooltip."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self._fig = Figure(figsize=(6, 4), dpi=100)
        self._ax = self._fig.add_subplot(111)
        super().__init__(self._fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._annotation = self._ax.annotate(
            "", xy=(0, 0), xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.9),
            fontsize=8,
            visible=False,
        )
        self._kde_line = None
        self._kde_fn = None
        self._sorted_swings: np.ndarray | None = None
        self.mpl_connect("motion_notify_event", self._on_mouse_move)

    def plot_swings(
        self,
        swings: np.ndarray,
        n_tourneys: int,
        n_simulations: int,
        xlabel: str = "Swing (buy-ins)",
        metric_label: str = "Swing",
        title_suffix: str = "",
    ) -> None:
        self._ax.clear()
        self._kde_fn = None
        self._kde_line = None
        self._sorted_swings = np.sort(swings)

        n_bins = min(200, max(50, int(np.sqrt(len(swings)))))

        self._ax.hist(
            swings, bins=n_bins, density=True,
            alpha=0.35, color="#4C72B0", edgecolor="none",
        )

        try:
            kde = gaussian_kde(swings, bw_method="scott")
            x_grid = np.linspace(float(swings.min()), float(swings.max()), 500)
            y_kde = kde(x_grid)
            self._kde_line, = self._ax.plot(
                x_grid, y_kde, color="#2855A0", linewidth=2.0,
            )
            self._kde_fn = kde
        except Exception:
            pass

        self._ax.axvline(0, color="#C44E52", linewidth=1.5, linestyle="--", label="Zero")

        mean_val = float(np.mean(swings))
        self._ax.axvline(
            mean_val, color="#55A868", linewidth=1.5, linestyle="-",
            label=f"Mean: {_fmt_thousands_2(mean_val)}",
        )

        ci_lo = float(np.percentile(swings, 2.5))
        ci_hi = float(np.percentile(swings, 97.5))
        self._ax.axvline(ci_lo, color="#8172B2", linewidth=1, linestyle=":", label="95% CI")
        self._ax.axvline(ci_hi, color="#8172B2", linewidth=1, linestyle=":")

        self._ax.xaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
        self._ax.set_xlabel(xlabel, fontsize=11)
        self._ax.set_ylabel("Density", fontsize=11)
        self._ax.set_title(
            f"{metric_label} Distribution  —  {n_tourneys:,} tourneys × {n_simulations:,} sims"
            + (f"  ({title_suffix})" if title_suffix else ""),
            fontsize=13, fontweight="bold", pad=12,
        )
        self._ax.legend(loc="upper right", fontsize=8)
        self._fig.subplots_adjust(top=0.88, bottom=0.13, left=0.12, right=0.96)

        self._annotation = self._ax.annotate(
            "", xy=(0, 0), xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.9),
            fontsize=8, visible=False,
        )
        self.draw()

    def _on_mouse_move(self, event) -> None:
        if event.inaxes != self._ax or self._kde_fn is None:
            if self._annotation.get_visible():
                self._annotation.set_visible(False)
                self.draw_idle()
            return
        x = event.xdata
        y = float(self._kde_fn(x))

        pct_left = 0.0
        if self._sorted_swings is not None and len(self._sorted_swings) > 0:
            idx = np.searchsorted(self._sorted_swings, x, side="right")
            pct_left = idx / len(self._sorted_swings) * 100.0

        self._annotation.xy = (x, y)
        self._annotation.set_text(
            f"x = {_fmt_thousands_2(x)}\n"
            f"← {pct_left:.2f}%  |  {100.0 - pct_left:.2f}% →"
        )
        self._annotation.set_visible(True)
        self.draw_idle()

    def clear_plot(self) -> None:
        self._ax.clear()
        self._kde_fn = None
        self._kde_line = None
        self._sorted_swings = None
        self._ax.set_title("No simulation results yet", fontsize=13, fontweight="bold", pad=12)
        self._fig.subplots_adjust(top=0.88, bottom=0.13, left=0.12, right=0.96)
        self.draw()


class ResultsPanel(QGroupBox):
    """Displays simulation results: stats, histogram, percentile table, summary."""

    CHART_MODES = ["BI", "BI/10k", "$"]
    PERCENTILE_STEPS = list(range(1, 100))

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Simulation Results", parent)

        self._buyin = 1.0
        self._last_result: SimulationResult | None = None
        self._last_portfolio: PortfolioResult | None = None
        self._is_portfolio = False
        self._last_chart_swings: np.ndarray | None = None
        self._last_chart_mode: str | None = None

        main_layout = QVBoxLayout(self)

        self._stats_label = QLabel("Run a simulation to see results.")
        self._stats_label.setWordWrap(True)
        self._stats_label.setTextFormat(Qt.TextFormat.RichText)
        main_layout.addWidget(self._stats_label)

        self._summary_cols_swing = [
            "Configuration", "Room", "Stake", "Formula", "N",
            "σ/tourney ($)", "σ total ($)",
            "σ/tourney (BI)", "σ total (BI)", "σ BI/10k",
        ]
        self._summary_cols_ev = [
            "Configuration", "Room", "Stake", "Formula", "N",
            "Mean ($)",
            "σ/tourney ($)", "σ total ($)",
            "σ/tourney (BI)", "σ total (BI)", "σ BI/10k",
        ]
        self._summary_cols = self._summary_cols_swing
        self._summary_table = QTableWidget(0, len(self._summary_cols))
        self._summary_table.setHorizontalHeaderLabels(self._summary_cols)
        self._summary_table.verticalHeader().setVisible(False)
        sh = self._summary_table.horizontalHeader()
        for i in range(len(self._summary_cols)):
            sh.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
        self._summary_table.setMaximumHeight(180)
        main_layout.addWidget(self._summary_table)

        export_row = QHBoxLayout()
        self._btn_export_csv = QPushButton("Export CSV")
        self._btn_export_csv.clicked.connect(lambda: self._export_table("csv"))
        export_row.addWidget(self._btn_export_csv)
        self._btn_export_xlsx = QPushButton("Export Excel")
        self._btn_export_xlsx.clicked.connect(lambda: self._export_table("xlsx"))
        export_row.addWidget(self._btn_export_xlsx)
        export_row.addStretch()
        main_layout.addLayout(export_row)

        chart_ctrl = QHBoxLayout()
        chart_ctrl.addWidget(QLabel("Chart metric:"))
        self._combo_chart = QComboBox()
        for mode in self.CHART_MODES:
            self._combo_chart.addItem(mode)
        self._combo_chart.setCurrentIndex(0)
        self._combo_chart.currentIndexChanged.connect(self._on_chart_mode_changed)
        chart_ctrl.addWidget(self._combo_chart)
        chart_ctrl.addStretch()
        main_layout.addLayout(chart_ctrl)

        mid_layout = QHBoxLayout()

        left_chart_col = QVBoxLayout()
        self._canvas = HistogramCanvas()
        self._canvas.clear_plot()
        left_chart_col.addWidget(self._canvas, stretch=1)

        chart_export_row = QHBoxLayout()
        self._btn_export_distribution = QPushButton("Export Distribution Data (CSV)")
        self._btn_export_distribution.clicked.connect(self._export_distribution_data_csv)
        chart_export_row.addWidget(self._btn_export_distribution)
        chart_export_row.addStretch()
        left_chart_col.addLayout(chart_export_row)
        mid_layout.addLayout(left_chart_col, stretch=3)

        self._pct_table = QTableWidget(len(self.PERCENTILE_STEPS), 2)
        self._pct_table.setHorizontalHeaderLabels(["Percentile", "Swing"])
        self._pct_table.verticalHeader().setVisible(False)
        header = self._pct_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._pct_table.setMaximumWidth(250)

        pct_right = QVBoxLayout()
        pct_right.addWidget(self._pct_table, stretch=1)
        self._btn_export_pct_xlsx = QPushButton("Export Percentiles Excel")
        self._btn_export_pct_xlsx.clicked.connect(self._export_percentiles_xlsx)
        pct_right.addWidget(self._btn_export_pct_xlsx)
        mid_layout.addLayout(pct_right, stretch=1)

        main_layout.addLayout(mid_layout)

    def set_buyin(self, buyin: float) -> None:
        self._buyin = buyin

    def show_result(self, result: SimulationResult) -> None:
        """Display result for a single setup simulation."""
        self._last_result = result
        self._last_portfolio = None
        self._is_portfolio = False
        self._ensure_summary_columns(result.metric_label)

        bi = result.buyin if result.buyin > 0 else (self._buyin if self._buyin > 0 else 1.0)
        self._populate_stats(result, bi)
        self._populate_summary_single(result, bi)
        self._refresh_chart()
        self._populate_percentiles(result, bi)

    def show_portfolio_result(self, portfolio: PortfolioResult) -> None:
        """Display combined portfolio result with per-entry breakdown."""
        self._last_result = portfolio.combined
        self._last_portfolio = portfolio
        self._is_portfolio = True
        self._ensure_summary_columns(portfolio.combined.metric_label)

        self._populate_stats_portfolio(portfolio)
        self._populate_summary_portfolio(portfolio)
        self._refresh_chart()
        self._populate_percentiles_portfolio(portfolio)

    def _populate_stats(self, result: SimulationResult, bi: float) -> None:
        ci95_lo_bi = result.ci_95[0] / bi
        ci95_hi_bi = result.ci_95[1] / bi
        ci99_lo_bi = result.ci_99[0] / bi
        ci99_hi_bi = result.ci_99[1] / bi
        mean_bi = result.mean_swing / bi

        total_pool_buyin = bi * result.n_tourneys if result.n_tourneys > 0 else 1.0
        sigma_bi_10k = result.sigma_simulated_portfolio / total_pool_buyin * 10_000
        swing_bi_10k = (result.mean_swing / total_pool_buyin * 10_000) if result.n_tourneys > 0 else 0

        mean_label = "swing" if result.metric_label == "Swing" else result.metric_label
        ppos_label = "swing" if result.metric_label == "Swing" else result.metric_label

        stats_html = (
            f"<table cellspacing='4'>"
            f"<tr><td colspan='3'><b>--- Analytical ---</b></td></tr>"
            f"<tr><td>σ per tourney:</td>"
            f"<td>${_fmt_thousands_2(result.sigma_analytical)}</td>"
            f"<td>({result.sigma_analytical / bi:.4f} BI)</td></tr>"
            f"<tr><td>σ portfolio ({_fmt_thousands_int(result.n_tourneys)}):</td>"
            f"<td>${_fmt_thousands_2(result.sigma_analytical_portfolio)}</td>"
            f"<td>({_fmt_thousands_2(result.sigma_analytical_portfolio / bi)} BI)</td></tr>"
            f"<tr><td colspan='3'><b>--- Simulated ---</b></td></tr>"
            f"<tr><td>σ per tourney:</td>"
            f"<td>${_fmt_thousands_2(result.sigma_simulated)}</td>"
            f"<td>({result.sigma_simulated / bi:.4f} BI)</td></tr>"
            f"<tr><td>σ portfolio ({_fmt_thousands_int(result.n_tourneys)}):</td>"
            f"<td>${_fmt_thousands_2(result.sigma_simulated_portfolio)}</td>"
            f"<td>({_fmt_thousands_2(result.sigma_simulated_portfolio / bi)} BI)</td></tr>"
            f"<tr><td>σ BI/10k:</td>"
            f"<td>{_fmt_thousands_2(sigma_bi_10k)} BI</td><td></td></tr>"
            f"<tr><td colspan='3'>&nbsp;</td></tr>"
            f"<tr><td>Mean {mean_label}:</td>"
            f"<td>${_fmt_thousands_2(result.mean_swing)}</td>"
            f"<td>({mean_bi:+.4f} BI)</td></tr>"
            f"<tr><td>95% CI:</td>"
            f"<td>[${_fmt_thousands_2(result.ci_95[0])}, ${_fmt_thousands_2(result.ci_95[1])}]</td>"
            f"<td>[{ci95_lo_bi:+,.2f}, {ci95_hi_bi:+,.2f}] BI</td></tr>"
            f"<tr><td>99% CI:</td>"
            f"<td>[${_fmt_thousands_2(result.ci_99[0])}, ${_fmt_thousands_2(result.ci_99[1])}]</td>"
            f"<td>[{ci99_lo_bi:+,.2f}, {ci99_hi_bi:+,.2f}] BI</td></tr>"
            f"<tr><td>P({ppos_label} &gt; 0):</td>"
            f"<td>{result.prob_positive:.2%}</td>"
            f"<td>P({ppos_label} &lt; 0): {result.prob_negative:.2%}</td></tr>"
            f"</table>"
        )
        self._stats_label.setText(stats_html)

    def _populate_stats_portfolio(self, pf: PortfolioResult) -> None:
        r = pf.combined
        mean_label = "swing" if r.metric_label == "Swing" else r.metric_label
        ppos_label = "swing" if r.metric_label == "Swing" else r.metric_label
        stats_html = (
            f"<table cellspacing='4'>"
            f"<tr><td colspan='3'><b>--- Portfolio (combined) ---</b></td></tr>"
            f"<tr><td colspan='3'><b>--- Analytical ---</b></td></tr>"
            f"<tr><td>σ portfolio:</td>"
            f"<td>${_fmt_thousands_2(r.sigma_analytical_portfolio)}</td><td></td></tr>"
            f"<tr><td colspan='3'><b>--- Simulated ---</b></td></tr>"
            f"<tr><td>σ portfolio:</td>"
            f"<td>${_fmt_thousands_2(r.sigma_simulated_portfolio)}</td><td></td></tr>"
            f"<tr><td colspan='3'>&nbsp;</td></tr>"
            f"<tr><td>Mean {mean_label}:</td>"
            f"<td>${_fmt_thousands_2(r.mean_swing)}</td><td></td></tr>"
            f"<tr><td>95% CI:</td>"
            f"<td>[${_fmt_thousands_2(r.ci_95[0])}, ${_fmt_thousands_2(r.ci_95[1])}]</td><td></td></tr>"
            f"<tr><td>99% CI:</td>"
            f"<td>[${_fmt_thousands_2(r.ci_99[0])}, ${_fmt_thousands_2(r.ci_99[1])}]</td><td></td></tr>"
            f"<tr><td>P({ppos_label} &gt; 0):</td>"
            f"<td>{r.prob_positive:.2%}</td>"
            f"<td>P({ppos_label} &lt; 0): {r.prob_negative:.2%}</td></tr>"
            f"</table>"
        )
        self._stats_label.setText(stats_html)

    def _populate_summary_single(self, result: SimulationResult, bi: float) -> None:
        config_name = result.config_name or "Single setup"
        sigma_pt_bi = result.sigma_simulated / bi
        total_pool_buyin = bi * result.n_tourneys if result.n_tourneys > 0 else 1.0
        sigma_bi_10k = result.sigma_simulated_portfolio / total_pool_buyin * 10_000
        mean_usd = result.mean_swing

        room_label = result.poker_room
        if result.reservation:
            room_label += f" ({result.reservation})"

        stake_native = result.buyin_native if result.buyin_native > 0 else result.buyin

        self._summary_table.setRowCount(1)
        self._summary_table.setItem(0, 0, self._ro(config_name))
        self._summary_table.setItem(0, 1, self._ro(room_label))
        self._summary_table.setItem(0, 2, self._ro(f"{stake_native:g}"))
        self._summary_table.setItem(0, 3, self._ro(result.ev_formula_name or "—"))
        self._summary_table.setItem(0, 4, self._ro(_fmt_thousands_int(result.n_tourneys)))
        if result.metric_label == "Swing":
            self._summary_table.setItem(0, 5, self._ro(_fmt_thousands_2(result.sigma_simulated)))
            self._summary_table.setItem(0, 6, self._ro(_fmt_thousands_2(result.sigma_simulated_portfolio)))
            self._summary_table.setItem(0, 7, self._ro(f"{sigma_pt_bi:.4f}"))
            self._summary_table.setItem(0, 8, self._ro(_fmt_thousands_2(result.sigma_simulated_portfolio / bi)))
            self._summary_table.setItem(0, 9, self._ro(_fmt_thousands_2(sigma_bi_10k)))
        else:
            self._summary_table.setItem(0, 5, self._ro(_fmt_thousands_2(mean_usd)))
            self._summary_table.setItem(0, 6, self._ro(_fmt_thousands_2(result.sigma_simulated)))
            self._summary_table.setItem(0, 7, self._ro(_fmt_thousands_2(result.sigma_simulated_portfolio)))
            self._summary_table.setItem(0, 8, self._ro(f"{sigma_pt_bi:.4f}"))
            self._summary_table.setItem(0, 9, self._ro(_fmt_thousands_2(result.sigma_simulated_portfolio / bi)))
            self._summary_table.setItem(0, 10, self._ro(_fmt_thousands_2(sigma_bi_10k)))

    def _populate_summary_portfolio(self, pf: PortfolioResult) -> None:
        n_entries = len(pf.entry_results)
        self._summary_table.setRowCount(n_entries + 1)

        for row, er in enumerate(pf.entry_results):
            bi = er.buyin_usd if er.buyin_usd > 0 else 1.0
            sigma_pt_bi = er.sigma_per_tourney_usd / bi
            sigma_port_bi = er.sigma_portfolio_usd / bi
            entry_pool_buyin = bi * er.n_tourneys if er.n_tourneys > 0 else 1.0
            sigma_bi_10k = er.sigma_portfolio_usd / entry_pool_buyin * 10_000
            mean_usd = float(np.mean(er.swings_usd)) if er.swings_usd is not None else 0.0

            room_label = er.poker_room
            if er.reservation:
                room_label += f" ({er.reservation})"

            stake_native = er.buyin_native if er.buyin_native > 0 else er.buyin_usd

            self._summary_table.setItem(row, 0, self._ro(er.name))
            self._summary_table.setItem(row, 1, self._ro(room_label))
            self._summary_table.setItem(row, 2, self._ro(f"{stake_native:g}"))
            self._summary_table.setItem(row, 3, self._ro(er.ev_formula_name))
            self._summary_table.setItem(row, 4, self._ro(_fmt_thousands_int(er.n_tourneys)))
            if pf.combined.metric_label == "Swing":
                self._summary_table.setItem(row, 5, self._ro(_fmt_thousands_2(er.sigma_per_tourney_usd)))
                self._summary_table.setItem(row, 6, self._ro(_fmt_thousands_2(er.sigma_portfolio_usd)))
                self._summary_table.setItem(row, 7, self._ro(f"{sigma_pt_bi:.4f}"))
                self._summary_table.setItem(row, 8, self._ro(_fmt_thousands_2(sigma_port_bi)))
                self._summary_table.setItem(row, 9, self._ro(_fmt_thousands_2(sigma_bi_10k)))
            else:
                self._summary_table.setItem(row, 5, self._ro(_fmt_thousands_2(mean_usd)))
                self._summary_table.setItem(row, 6, self._ro(_fmt_thousands_2(er.sigma_per_tourney_usd)))
                self._summary_table.setItem(row, 7, self._ro(_fmt_thousands_2(er.sigma_portfolio_usd)))
                self._summary_table.setItem(row, 8, self._ro(f"{sigma_pt_bi:.4f}"))
                self._summary_table.setItem(row, 9, self._ro(_fmt_thousands_2(sigma_port_bi)))
                self._summary_table.setItem(row, 10, self._ro(_fmt_thousands_2(sigma_bi_10k)))

        r = pf.combined
        total_row = n_entries

        weighted_bi_sum = sum(er.buyin_usd * er.n_tourneys for er in pf.entry_results)
        avg_bi = weighted_bi_sum / pf.total_n_tourneys if pf.total_n_tourneys > 0 else 1.0

        sigma_port_bi = r.sigma_simulated_portfolio / avg_bi if avg_bi > 0 else 0.0
        sigma_pt_bi = sigma_port_bi / np.sqrt(pf.total_n_tourneys) if pf.total_n_tourneys > 0 else 0.0
        sigma_bi_10k = r.sigma_simulated_portfolio / weighted_bi_sum * 10_000 if weighted_bi_sum > 0 else 0.0

        self._summary_table.setItem(total_row, 0, self._ro_bold("PORTFOLIO TOTAL"))
        for c in range(1, 4):
            self._summary_table.setItem(total_row, c, self._ro_bold("—"))
        self._summary_table.setItem(total_row, 4, self._ro_bold(_fmt_thousands_int(pf.total_n_tourneys)))
        if r.metric_label == "Swing":
            self._summary_table.setItem(total_row, 5, self._ro_bold("—"))
            self._summary_table.setItem(total_row, 6, self._ro_bold(_fmt_thousands_2(r.sigma_simulated_portfolio)))
            self._summary_table.setItem(total_row, 7, self._ro_bold("—"))
            self._summary_table.setItem(total_row, 8, self._ro_bold(_fmt_thousands_2(sigma_port_bi)))
            self._summary_table.setItem(total_row, 9, self._ro_bold(_fmt_thousands_2(sigma_bi_10k)))
        else:
            self._summary_table.setItem(total_row, 5, self._ro_bold(_fmt_thousands_2(r.mean_swing)))
            self._summary_table.setItem(total_row, 6, self._ro_bold("—"))
            self._summary_table.setItem(total_row, 7, self._ro_bold(_fmt_thousands_2(r.sigma_simulated_portfolio)))
            self._summary_table.setItem(total_row, 8, self._ro_bold("—"))
            self._summary_table.setItem(total_row, 9, self._ro_bold(_fmt_thousands_2(sigma_port_bi)))
            self._summary_table.setItem(total_row, 10, self._ro_bold(_fmt_thousands_2(sigma_bi_10k)))

    def _ensure_summary_columns(self, metric_label: str) -> None:
        desired = self._summary_cols_swing if metric_label == "Swing" else self._summary_cols_ev
        if desired == self._summary_cols:
            return
        self._summary_cols = desired
        self._summary_table.setColumnCount(len(desired))
        self._summary_table.setHorizontalHeaderLabels(desired)
        sh = self._summary_table.horizontalHeader()
        for i in range(len(desired)):
            sh.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
        # Clear stale content when shape changes
        self._summary_table.setRowCount(0)

    def _on_chart_mode_changed(self) -> None:
        self._refresh_chart()

    def _refresh_chart(self) -> None:
        result = self._last_result
        if result is None:
            return

        mode = self.CHART_MODES[self._combo_chart.currentIndex()]

        if mode == "$":
            swings = result.swings
            xlabel = f"{result.metric_label} ($)"
        elif mode == "BI":
            bi = self._get_effective_buyin()
            swings = result.swings / bi
            xlabel = f"{result.metric_label} (BI)"
        else:  # BI/10k
            bi = self._get_effective_buyin()
            n = result.n_tourneys if result.n_tourneys > 0 else 1
            total_pool_buyin = bi * n
            swings = result.swings / total_pool_buyin * 10_000
            xlabel = f"{result.metric_label} (BI/10k)"

        self._last_chart_swings = np.asarray(swings, dtype=np.float64).copy()
        self._last_chart_mode = mode

        self._canvas.plot_swings(
            swings, result.n_tourneys, result.n_simulations,
            xlabel=xlabel, metric_label=result.metric_label, title_suffix=mode,
        )

        pct_keys = self.PERCENTILE_STEPS
        self._pct_table.setRowCount(len(pct_keys))
        for row, pct in enumerate(pct_keys):
            val = float(np.percentile(swings, pct))
            pct_item = QTableWidgetItem(f"{pct:.0f}%")
            pct_item.setFlags(pct_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            pct_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._pct_table.setItem(row, 0, pct_item)
            val_item = QTableWidgetItem(_fmt_thousands_2(val))
            val_item.setFlags(val_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            val_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._pct_table.setItem(row, 1, val_item)

        self._pct_table.setHorizontalHeaderLabels(["Percentile", f"{result.metric_label} ({mode})"])

    def _get_effective_buyin(self) -> float:
        """Return the effective buyin for chart conversion.

        For portfolio: weighted average buyin across entries.
        For single setup: the setup's buyin.
        """
        if self._is_portfolio and self._last_portfolio is not None:
            pf = self._last_portfolio
            total_n = pf.total_n_tourneys
            if total_n > 0:
                return sum(er.buyin_usd * er.n_tourneys for er in pf.entry_results) / total_n
            return 1.0
        r = self._last_result
        if r is not None and r.buyin > 0:
            return r.buyin
        return self._buyin if self._buyin > 0 else 1.0

    def _populate_percentiles(self, result: SimulationResult, bi: float) -> None:
        pass  # handled by _refresh_chart

    def _populate_percentiles_portfolio(self, pf: PortfolioResult) -> None:
        pass  # handled by _refresh_chart

    def _export_table(self, fmt: str) -> None:
        """Export summary table to CSV or XLSX."""
        rows = self._summary_table.rowCount()
        cols = self._summary_table.columnCount()
        if rows == 0:
            return

        headers = []
        for c in range(cols):
            h = self._summary_table.horizontalHeaderItem(c)
            headers.append(h.text() if h else f"Col{c}")

        data: list[list[str]] = []
        for r in range(rows):
            row_data = []
            for c in range(cols):
                item = self._summary_table.item(r, c)
                val = item.text() if item else ""
                val = val.replace("\u00a0", "").replace(" ", "").lstrip("$")
                row_data.append(val)
            data.append(row_data)

        if fmt == "csv":
            path, _ = QFileDialog.getSaveFileName(
                self, "Export CSV", "portfolio_results.csv", "CSV Files (*.csv)"
            )
            if not path:
                return
            import csv
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(headers)
                writer.writerows(data)

        elif fmt == "xlsx":
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Excel", "portfolio_results.xlsx", "Excel Files (*.xlsx)"
            )
            if not path:
                return
            try:
                import openpyxl
                wb = openpyxl.Workbook()
                ws = wb.active
                ws.title = "Results"
                ws.append(headers)
                for row_data in data:
                    ws.append(row_data)
                for col_cells in ws.columns:
                    max_len = max(len(str(c.value or "")) for c in col_cells)
                    ws.column_dimensions[col_cells[0].column_letter].width = max(max_len + 2, 12)
                wb.save(path)
            except ImportError:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self, "Missing dependency",
                    "Install openpyxl to export Excel:\npip install openpyxl",
                )

    def _export_percentiles_xlsx(self) -> None:
        """Export the currently visible percentiles table to XLSX."""
        rows = self._pct_table.rowCount()
        cols = self._pct_table.columnCount()
        if rows == 0 or cols == 0:
            return

        headers = []
        for c in range(cols):
            h = self._pct_table.horizontalHeaderItem(c)
            headers.append(h.text() if h else f"Col{c}")

        data: list[list[str]] = []
        for r in range(rows):
            row_data = []
            for c in range(cols):
                item = self._pct_table.item(r, c)
                row_data.append(item.text() if item else "")
            data.append(row_data)

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Percentiles Excel", "percentiles.xlsx", "Excel Files (*.xlsx)"
        )
        if not path:
            return

        try:
            import openpyxl
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Percentiles"
            ws.append(headers)
            for row_data in data:
                ws.append(row_data)
            for col_cells in ws.columns:
                max_len = max(len(str(c.value or "")) for c in col_cells)
                ws.column_dimensions[col_cells[0].column_letter].width = max(max_len + 2, 12)
            wb.save(path)
        except ImportError:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Missing dependency",
                "Install openpyxl to export Excel:\npip install openpyxl",
            )

    def _export_distribution_data_csv(self) -> None:
        """Export exact chart data required to reproduce current distribution."""
        result = self._last_result
        swings = self._last_chart_swings
        mode = self._last_chart_mode
        if result is None or swings is None or mode is None or swings.size == 0:
            return

        default_name = f"distribution_{mode.replace('/', '_')}_{result.n_tourneys}_{result.n_simulations}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Distribution Data (CSV)",
            default_name,
            "CSV Files (*.csv)",
        )
        if not path:
            return

        n_bins = min(200, max(50, int(np.sqrt(len(swings)))))
        percentiles = [1, 2.5, 5, 25, 50, 75, 95, 97.5, 99]
        file_path = Path(path)

        with file_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")

            writer.writerow(["meta_key", "meta_value"])
            writer.writerow(["metric", result.metric_label])
            writer.writerow(["chart_mode", mode])
            writer.writerow(["n_tourneys", result.n_tourneys])
            writer.writerow(["n_simulations", result.n_simulations])
            writer.writerow(["n_points", int(swings.size)])
            writer.writerow(["hist_n_bins", n_bins])
            writer.writerow(["mean", float(np.mean(swings))])
            writer.writerow(["std", float(np.std(swings))])
            writer.writerow(["ci_95_lo", float(np.percentile(swings, 2.5))])
            writer.writerow(["ci_95_hi", float(np.percentile(swings, 97.5))])
            writer.writerow([])

            writer.writerow(["percentile", "value"])
            for p in percentiles:
                writer.writerow([p, float(np.percentile(swings, p))])
            writer.writerow([])

            writer.writerow([result.metric_label.lower()])
            for val in swings:
                writer.writerow([float(val)])

    def clear_results(self) -> None:
        self._stats_label.setText("Run a simulation to see results.")
        self._canvas.clear_plot()
        self._pct_table.clearContents()
        self._summary_table.setRowCount(0)
        self._last_result = None
        self._last_portfolio = None
        self._last_chart_swings = None
        self._last_chart_mode = None

    @staticmethod
    def _ro(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        return item

    @staticmethod
    def _ro_bold(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        return item

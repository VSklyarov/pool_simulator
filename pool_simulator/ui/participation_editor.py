"""Editable table for multiplier participation rates and per-multiplier ChipEV."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QGroupBox,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pool_simulator.core.models import Multiplier


class ParticipationEditor(QGroupBox):
    """Table showing multiplier details with editable participation and ChipEV columns."""

    participation_changed = Signal()
    chip_ev_changed = Signal()

    COLUMNS = ["Multiplier", "W1", "W23", "Probability", "Stack", "ChipEV", "Participation"]

    COL_MULT = 0
    COL_W1 = 1
    COL_W23 = 2
    COL_PROB = 3
    COL_STACK = 4
    COL_CHIPEV = 5
    COL_PART = 6

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Multiplier Participation", parent)
        self._multipliers: list[Multiplier] = []
        self._part_spinboxes: list[QDoubleSpinBox] = []
        self._cev_spinboxes: list[QDoubleSpinBox] = []
        self._has_variable_stacks = False
        self._min_stack = 0

        self._table = QTableWidget(0, len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(self.COLUMNS)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.verticalHeader().setVisible(False)

        header = self._table.horizontalHeader()
        for i in range(len(self.COLUMNS)):
            if i in (self.COL_CHIPEV, self.COL_PART):
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Fixed)
                self._table.setColumnWidth(i, 100)
            else:
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)

        layout = QVBoxLayout(self)
        layout.addWidget(self._table)

    def set_multipliers(
        self,
        multipliers: list[Multiplier],
        default_chip_evs: dict[int, float] | None = None,
        has_variable_stacks: bool = False,
    ) -> None:
        self._multipliers = multipliers
        self._part_spinboxes.clear()
        self._cev_spinboxes.clear()
        self._has_variable_stacks = has_variable_stacks
        self._table.setRowCount(len(multipliers))

        if multipliers:
            self._min_stack = min(m.stack_size for m in multipliers)
        else:
            self._min_stack = 0

        if default_chip_evs is None:
            default_chip_evs = {}

        for row, m in enumerate(multipliers):
            self._table.setItem(row, self.COL_MULT, self._readonly_item(f"x{m.multiplier_value:g}"))
            self._table.setItem(row, self.COL_W1, self._readonly_item(f"{m.w1:g}"))
            self._table.setItem(row, self.COL_W23, self._readonly_item(f"{m.w23:g}"))
            self._table.setItem(row, self.COL_PROB, self._readonly_item(self._format_prob(m.probability)))
            self._table.setItem(row, self.COL_STACK, self._readonly_item(str(m.stack_size)))

            cev_default = default_chip_evs.get(m.stack_size, 0.0)
            cev_spin = QDoubleSpinBox()
            cev_spin.setRange(-2000.0, 2000.0)
            cev_spin.setSingleStep(1.0)
            cev_spin.setDecimals(2)
            cev_spin.setMinimumWidth(90)
            cev_spin.setValue(cev_default)
            cev_spin.valueChanged.connect(self._on_chip_ev_changed)
            self._table.setCellWidget(row, self.COL_CHIPEV, cev_spin)
            self._cev_spinboxes.append(cev_spin)

            spin = QDoubleSpinBox()
            spin.setRange(m.min_participation, m.max_participation)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            spin.setMinimumWidth(90)
            spin.setValue(m.default_participation)
            spin.valueChanged.connect(self._on_value_changed)
            self._table.setCellWidget(row, self.COL_PART, spin)
            self._part_spinboxes.append(spin)

        self._table.resizeRowsToContents()

    def get_participations(self) -> list[float]:
        return [spin.value() for spin in self._part_spinboxes]

    def get_chip_evs_per_multiplier(self) -> list[float]:
        """Return per-multiplier ChipEV values (one per row, in order)."""
        return [spin.value() for spin in self._cev_spinboxes]

    def get_chip_evs_dict(self) -> dict[int, float]:
        """Return {stack_size: chip_ev} using the first occurrence of each stack."""
        result: dict[int, float] = {}
        for m, spin in zip(self._multipliers, self._cev_spinboxes):
            if m.stack_size not in result:
                result[m.stack_size] = spin.value()
        return result

    def set_participations(self, values: list[float]) -> None:
        """Set participation values programmatically."""
        for i, val in enumerate(values):
            if i < len(self._part_spinboxes):
                self._part_spinboxes[i].blockSignals(True)
                self._part_spinboxes[i].setValue(val)
                self._part_spinboxes[i].blockSignals(False)
        self.participation_changed.emit()

    def set_chip_evs(self, chip_evs: dict[int, float]) -> None:
        """Set per-multiplier ChipEV from {stack_size: value}."""
        for m, spin in zip(self._multipliers, self._cev_spinboxes):
            val = chip_evs.get(m.stack_size, 0.0)
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
        self.chip_ev_changed.emit()

    def set_chip_evs_from_global(self, global_value: float) -> None:
        """Update per-multiplier ChipEV based on the global input value.

        Fixed stacks: global_value goes to all multipliers.
        Variable stacks: global_value goes to min-stack multipliers;
        others are scaled proportionally.
        """
        if not self._multipliers:
            return
        for m, spin in zip(self._multipliers, self._cev_spinboxes):
            spin.blockSignals(True)
            if not self._has_variable_stacks or self._min_stack == 0:
                spin.setValue(global_value)
            else:
                spin.setValue(global_value * m.stack_size / self._min_stack)
            spin.blockSignals(False)
        self.chip_ev_changed.emit()

    def _on_value_changed(self) -> None:
        self.participation_changed.emit()

    def _on_chip_ev_changed(self) -> None:
        self.chip_ev_changed.emit()

    @staticmethod
    def _readonly_item(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        return item

    @staticmethod
    def _format_prob(p: float) -> str:
        if p >= 0.01:
            return f"{p:.4f}"
        elif p >= 0.0001:
            return f"{p:.6f}"
        else:
            return f"{p:.2e}"

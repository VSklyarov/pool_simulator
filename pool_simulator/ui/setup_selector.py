"""Cascading combo-box widget for selecting a tournament setup."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QWidget,
)

from pool_simulator.core.models import Setup


class SetupSelector(QGroupBox):
    """Four cascading dropdowns: Room -> Reservation -> Type -> Buy-in."""

    setup_changed = Signal(object)  # emits the selected Setup or None

    def __init__(self, setups: list[Setup], parent: QWidget | None = None) -> None:
        super().__init__("Setup Selection", parent)
        self._setups = setups
        self._current_setup: Setup | None = None

        self._combo_room = QComboBox()
        self._combo_reservation = QComboBox()
        self._combo_type = QComboBox()
        self._combo_buyin = QComboBox()

        layout = QFormLayout(self)
        layout.addRow("Poker Room:", self._combo_room)
        layout.addRow("Reservation:", self._combo_reservation)
        layout.addRow("Type:", self._combo_type)
        layout.addRow("Buy-in:", self._combo_buyin)

        self._combo_room.currentTextChanged.connect(self._on_room_changed)
        self._combo_reservation.currentTextChanged.connect(self._on_reservation_changed)
        self._combo_type.currentTextChanged.connect(self._on_type_changed)
        self._combo_buyin.currentTextChanged.connect(self._on_buyin_changed)

        self._populate_rooms()

    @property
    def current_setup(self) -> Setup | None:
        return self._current_setup

    def select_setup(self, setup: Setup) -> None:
        """Programmatically select a setup by setting all dropdowns."""
        room_idx = self._combo_room.findText(setup.poker_room)
        if room_idx < 0:
            return
        self._combo_room.setCurrentIndex(room_idx)

        res_idx = self._combo_reservation.findText(setup.reservation)
        if res_idx >= 0:
            self._combo_reservation.setCurrentIndex(res_idx)

        type_idx = self._combo_type.findText(setup.tournament_type)
        if type_idx >= 0:
            self._combo_type.setCurrentIndex(type_idx)

        buyin_text = self._format_buyin(setup.buyin)
        buyin_idx = self._combo_buyin.findText(buyin_text)
        if buyin_idx >= 0:
            self._combo_buyin.setCurrentIndex(buyin_idx)

    def _populate_rooms(self) -> None:
        rooms = sorted({s.poker_room for s in self._setups})
        self._combo_room.blockSignals(True)
        self._combo_room.clear()
        self._combo_room.addItems(rooms)
        self._combo_room.blockSignals(False)
        if rooms:
            self._combo_room.setCurrentIndex(0)
            self._on_room_changed(rooms[0])

    def _filtered(
        self,
        room: str | None = None,
        reservation: str | None = None,
        ttype: str | None = None,
    ) -> list[Setup]:
        result = self._setups
        if room:
            result = [s for s in result if s.poker_room == room]
        if reservation:
            result = [s for s in result if s.reservation == reservation]
        if ttype:
            result = [s for s in result if s.tournament_type == ttype]
        return result

    def _on_room_changed(self, room: str) -> None:
        filtered = self._filtered(room=room)
        reservations = sorted({s.reservation for s in filtered})

        self._combo_reservation.blockSignals(True)
        self._combo_reservation.clear()
        self._combo_reservation.addItems(reservations)
        self._combo_reservation.blockSignals(False)

        if reservations:
            self._combo_reservation.setCurrentIndex(0)
            self._on_reservation_changed(reservations[0])
        else:
            self._clear_downstream("reservation")

    def _on_reservation_changed(self, reservation: str) -> None:
        room = self._combo_room.currentText()
        filtered = self._filtered(room=room, reservation=reservation)
        types = sorted({s.tournament_type for s in filtered})

        self._combo_type.blockSignals(True)
        self._combo_type.clear()
        self._combo_type.addItems(types)
        self._combo_type.blockSignals(False)

        if types:
            self._combo_type.setCurrentIndex(0)
            self._on_type_changed(types[0])
        else:
            self._clear_downstream("type")

    def _on_type_changed(self, ttype: str) -> None:
        room = self._combo_room.currentText()
        reservation = self._combo_reservation.currentText()
        filtered = self._filtered(room=room, reservation=reservation, ttype=ttype)
        buyins = sorted({s.buyin for s in filtered})

        self._combo_buyin.blockSignals(True)
        self._combo_buyin.clear()
        self._combo_buyin.addItems([self._format_buyin(b) for b in buyins])
        self._combo_buyin.blockSignals(False)

        if buyins:
            self._combo_buyin.setCurrentIndex(0)
            self._on_buyin_changed(self._format_buyin(buyins[0]))
        else:
            self._clear_downstream("buyin")

    def _on_buyin_changed(self, buyin_text: str) -> None:
        if not buyin_text:
            self._current_setup = None
            self.setup_changed.emit(None)
            return

        buyin = self._parse_buyin(buyin_text)
        room = self._combo_room.currentText()
        reservation = self._combo_reservation.currentText()
        ttype = self._combo_type.currentText()

        matches = [
            s
            for s in self._setups
            if s.poker_room == room
            and s.reservation == reservation
            and s.tournament_type == ttype
            and abs(s.buyin - buyin) < 0.01
        ]
        self._current_setup = matches[0] if matches else None
        self.setup_changed.emit(self._current_setup)

    def _clear_downstream(self, level: str) -> None:
        if level in ("reservation", "room"):
            self._combo_type.clear()
        if level in ("reservation", "room", "type"):
            self._combo_buyin.clear()
        self._current_setup = None
        self.setup_changed.emit(None)

    @staticmethod
    def _format_buyin(buyin: float) -> str:
        if buyin == int(buyin):
            return str(int(buyin))
        return f"{buyin:.2f}"

    @staticmethod
    def _parse_buyin(text: str) -> float:
        return float(text.replace(",", "."))

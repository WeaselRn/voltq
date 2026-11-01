"""OCPP-like message -> VoltQ event adapter (SIMULATED telemetry only).

This shows where a real integration would plug in: an OCPP 1.6 / 2.0.1 gateway would receive
StatusNotification, ChangeAvailability or SetChargingProfile frames from chargers and hand them to
`event_from_message`. VoltQ never opens a connection to a charger; the dashboard feeds these
structures from the simulator. Field names follow OCPP loosely and are simplified.

    OCPP StatusNotification (status Faulted / Unavailable)  -> StationOutage
    OCPP ChangeAvailability (type Inoperative)              -> StationOutage
    OCPP SetChargingProfile (station-wide limit in kW)       -> PowerReduction
    OCPI-style tariff update                                 -> TariffChange
Routing services (travel time) and grid telemetry would feed `NewArrival` coordinates and the
station power cap respectively; neither is implemented here.
"""
from __future__ import annotations

from ..recovery.events import Event, PowerReduction, StationOutage, TariffChange
from ..recovery.state import NetworkState

DOWN_STATES = {"Faulted", "Unavailable", "Inoperative"}


def _station_id(state: NetworkState, name: str) -> int:
    for s in state.scn.stations:
        if s.name == name:
            return s.station_id
    raise KeyError(f"unknown station {name!r}")


def event_from_message(msg: dict, state: NetworkState, slot: int) -> Event:
    """Translate one simulated charger / grid message into a VoltQ event at `slot`."""
    kind = msg.get("action")
    if kind in ("StatusNotification", "ChangeAvailability"):
        status = msg.get("status") or msg.get("type")
        if status not in DOWN_STATES:
            raise ValueError(f"status {status!r} does not take the station out of service")
        return StationOutage(slot, _station_id(state, msg["stationId"]))
    if kind == "SetChargingProfile":
        sid = _station_id(state, msg["stationId"])
        st = state.scn.stations[sid]
        nominal = st.ports * st.power_kw
        return PowerReduction(slot, sid, max(0.05, min(1.0, float(msg["limit_kw"]) / nominal)))
    if kind == "TariffUpdate":
        return TariffChange(slot, int(msg["from_slot"]), int(msg["to_slot"]), float(msg["multiplier"]))
    raise ValueError(f"unsupported message {kind!r}")

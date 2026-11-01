"""Residual port and grid-power capacity given a set of fixed sessions."""
from __future__ import annotations

from ..models import Option, Scenario, grid_limit_kw


class Capacity:
    """Tracks, per (station, slot), ports and kW still free.

    Port rule:  concurrent sessions <= ports.      Power rule:  sum of session kW <= G_{s,t}.
    """

    def __init__(self, scn: Scenario, fixed: dict[int, Option] | None = None):
        self.stations = {s.station_id: s for s in scn.stations}
        self.used: dict[tuple[int, int], int] = {}
        self.kw: dict[tuple[int, int], float] = {}
        for o in (fixed or {}).values():
            self.add(o)

    def add(self, o: Option) -> None:
        for t in o.slots:
            k = (o.station_id, t)
            self.used[k] = self.used.get(k, 0) + 1
            self.kw[k] = self.kw.get(k, 0.0) + o.power_kw

    def remove(self, o: Option) -> None:
        for t in o.slots:
            k = (o.station_id, t)
            self.used[k] -= 1
            self.kw[k] -= o.power_kw

    def ports_free(self, s: int, t: int) -> int:
        return self.stations[s].ports - self.used.get((s, t), 0)

    def power_free(self, s: int, t: int) -> float:
        return grid_limit_kw(self.stations[s], t) - self.kw.get((s, t), 0.0)

    def fits(self, o: Option) -> bool:
        return all(self.ports_free(o.station_id, t) >= 1
                   and self.power_free(o.station_id, t) >= o.power_kw - 1e-9 for t in o.slots)

"""Discrete physical control levels; reward weights never redefine intensity."""
from dataclasses import dataclass
import numpy as np

CELLS = ("5g", "4g1", "4g2")
CIO_STEPS = (1, 2, 3, 4, 5, 6)
CIO_MIN, CIO_MAX = -6.0, 6.0
TTT_LEVELS = (40, 64, 128, 256, 512, 1024)
HYS_LEVELS = (0, 1, 2, 3, 4, 6)


@dataclass(frozen=True)
class Action:
    app: str
    cell: str = ""
    delta_db: int = 0
    ttt_ms: int = 0
    hys_db: int = 0

    def updates(self, parameters):
        if self.app == "hold":
            return {}
        if self.app == "lb":
            key = "cio_" + self.cell
            value = float(np.clip(parameters[key] + self.delta_db, CIO_MIN, CIO_MAX))
            return {} if abs(value - parameters[key]) < 1e-8 else {key: value}
        proposed = {"ttt_ms": float(self.ttt_ms), "hys_db": float(self.hys_db)}
        return {k: v for k, v in proposed.items() if abs(v - parameters[k]) > 1e-8}

    def label(self):
        if self.app == "hold":
            return "hold"
        if self.app == "lb":
            return f"CIO[{self.cell}] {self.delta_db:+d} dB"
        return f"TTT={self.ttt_ms} ms; HYS={self.hys_db} dB"


def catalog(app):
    if app == "lb":
        return [Action("hold")] + [Action("lb", c, s * m)
            for c in CELLS for s in (-1, 1) for m in CIO_STEPS]
    if app == "mo":
        return [Action("hold")] + [Action("mo", ttt_ms=t, hys_db=h)
            for t in TTT_LEVELS for h in HYS_LEVELS]
    raise ValueError(f"Unknown xApp {app!r}")


def valid_mask(actions, parameters):
    """Keep hold; mask clipped no-ops and duplicate *executed* control vectors."""
    seen = set()
    mask = []
    for action in actions:
        identity = tuple(sorted(action.updates(parameters).items()))
        mask.append(identity not in seen)
        seen.add(identity)
    return np.asarray(mask, dtype=bool)


def movement_cost(app, old, updates):
    if not updates:
        return 0.0
    if app == "lb":
        return sum(abs(v - old[k]) / 6 for k, v in updates.items())
    return (abs(updates.get("ttt_ms", old["ttt_ms"]) - old["ttt_ms"]) / 984
            + abs(updates.get("hys_db", old["hys_db"]) - old["hys_db"]) / 6) / 2


def describe_catalog(app):
    return [{"id": i, "label": a.label(), **a.__dict__} for i, a in enumerate(catalog(app))]

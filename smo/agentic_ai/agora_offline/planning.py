"""Bounded candidate construction; no predictions or future data are read here."""

from copy import deepcopy
import re
from .contracts import CELLS, DOMAINS, ContractError


def parse_deterministic(prompt, current, horizon=180):
    """An explicitly labelled parser for demos, not a simulated LLM."""
    text = prompt.lower()
    cell_match = re.search(r"5g|4g1|4g2", text)
    cell = cell_match.group(0) if cell_match else "5g"
    kind_match = re.search(r"txp|ret|cio|ttt|hys", text)
    if not kind_match:
        raise ContractError(
            "Deterministic mode requires an explicit TxP, RET, CIO, TTT or HYS control name"
        )
    kind = kind_match.group(0)
    key = {"ttt": "ttt_ms", "hys": "hys_db"}.get(kind, kind + "_" + cell)
    tail = text[kind_match.end() :]
    transition = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(?:dbm|db|ms|deg|도)?\s*(?:에서|→|->|to)\s*(-?\d+(?:\.\d+)?)", tail
    )
    expected = {}
    if transition:
        expected[key], value = map(float, transition.groups())
    else:
        value_match = re.search(r"(?:=|을|를|to)?\s*(-?\d+(?:\.\d+)?)", tail)
        if not value_match:
            raise ContractError("Use an explicit requested value, e.g. Cell 5G TxP 43 -> 33")
        value = float(value_match.group(1))
    durations = list(re.finditer(r"(\d+(?:\.\d+)?)\s*(분|minutes?|mins?|초|seconds?|secs?)", text))
    durations = [m for m in durations if not re.match(r"\s*(?:뒤|후|later)", text[m.end() :])]
    durations = [
        m
        for m in durations
        if not re.match(r"\s*(?:관측|이력|데이터|history|observ)", text[m.end() :])
        and not re.search(
            r"(?:최근|과거|지난|last|past)\s*$", text[max(0, m.start() - 20) : m.start()]
        )
    ]
    durations.sort(
        key=lambda m: not bool(
            re.match(r"\s*(?:동안|간|유지)", text[m.end() :])
            or re.search(r"for\s*$", text[max(0, m.start() - 10) : m.start()])
        )
    )
    duration = durations[0] if durations else None
    if duration:
        horizon = float(duration.group(1)) * (
            60 if duration.group(2).startswith(("분", "min")) else 1
        )
        if horizon.is_integer():
            horizon = int(horizon)
    delay_match = re.search(r"(\d+)\s*초\s*(?:뒤|후)", text)
    changes = {key: value}
    controls = list(re.finditer(r"txp|ret|cio|ttt|hys", text))
    for i, match in enumerate(controls[1:], 1):
        control = match.group(0)
        local = text[match.end() : controls[i + 1].start() if i + 1 < len(controls) else len(text)]
        transition = re.match(
            r"\s*(-?\d+(?:\.\d+)?)\s*(?:dbm|db|ms|deg|도)?\s*(?:에서|→|->|to)\s*(-?\d+(?:\.\d+)?)",
            local,
        )
        direct = re.match(r"\s*(?:=|을|를|to)?\s*(-?\d+(?:\.\d+)?)", local)
        if not transition and not direct:
            continue
        preceding = re.findall(r"5g|4g1|4g2", text[: match.start()])
        target_cell = preceding[-1] if preceding else cell
        other_key = {"ttt": "ttt_ms", "hys": "hys_db"}.get(control, control + "_" + target_cell)
        if transition:
            expected[other_key], changes[other_key] = map(float, transition.groups())
        else:
            changes[other_key] = float(direct.group(1))
    objective = {
        "ret": "coverage",
        "cio": "load_balancing",
        "ttt": "handover_control",
        "hys": "handover_control",
    }.get(kind)
    if kind == "txp":
        before = current.get(key, expected.get(key))
        if before is None:
            raise ContractError(
                "TxP intent requires the current setting or an explicit before -> after value"
            )
        objective = "energy_saving" if value < before else "throughput"
    if kind == "ret" and ("throughput" in text or "처리량" in text):
        objective = "throughput"
    return dict(
        changes=changes,
        expected_current=expected,
        horizon_s=horizon,
        delay_s=int(delay_match.group(1)) if delay_match else 0,
        objective=objective,
        target_fraction=0.05 if kind == "txp" else 0.20,
        interpretation="Explicit deterministic parser; structured editor owns KPI goals and thresholds when supplied",
    )


def alternatives(
    requested,
    current,
    axis="all",
    base=None,
    values=None,
    cells=None,
    delays_s=None,
    fractions=None,
):
    base = deepcopy(base or requested)
    base_changes = base.get("changes", {})
    default_delays = [0, 30, 60, 90]
    result = []

    def add(ch, delay, axes):
        if ch:
            result.append(
                {"changes": ch, "delay_s": delay, "horizon_s": requested["horizon_s"], "axes": axes}
            )

    if axis in ("all", "time"):
        for delay in delays_s if delays_s is not None else default_delays[1:]:
            add(base_changes, delay, ["time"])
    if axis in ("all", "intensity"):
        if values is not None:
            if len(base_changes) != 1:
                raise ContractError(
                    "Absolute values require one changed control; use fractions for compound actions"
                )
            key = next(iter(base_changes))
            for value in values:
                add({key: value}, base.get("delay_s", 0), ["intensity"])
        else:
            for fraction in fractions if fractions is not None else [0.3, 0.6, 0.8]:
                ch = {}
                for key, target in base_changes.items():
                    desired = current[key] + fraction * (target - current[key])
                    if key.startswith("txp_"):
                        value = round(desired, 2)
                    else:
                        grid = [
                            v
                            for v in DOMAINS[key]
                            if (v - current[key]) * (target - current[key]) > 0
                            and abs(v - current[key]) <= abs(target - current[key])
                        ]
                        if not grid:
                            continue
                        value = min(grid, key=lambda v: abs(v - desired))
                    ch[key] = value
                add(ch, base.get("delay_s", 0), ["intensity"])
    if axis in ("all", "cell", "combined"):
        targets = cells if cells is not None else list(CELLS)
        for target_cell in targets:
            relocated = {}
            for key, target in base_changes.items():
                kind = key.split("_")[0]
                if kind in ("ttt", "hys"):
                    relocated[key] = target
                    continue
                new_key = kind + "_" + target_cell
                if new_key in relocated:
                    raise ContractError(
                        "Cannot collapse multiple requested cell controls onto one cell"
                    )
                delta = target - current[key]
                shifted = max(
                    min(current[new_key] + delta, DOMAINS[new_key][-1]), DOMAINS[new_key][0]
                )
                if kind != "txp":
                    grid = [
                        v
                        for v in DOMAINS[new_key]
                        if (v - current[new_key]) * delta > 0
                        and abs(v - current[new_key]) <= abs(delta)
                    ]
                    if not grid:
                        continue
                    shifted = min(grid, key=lambda v: abs(v - shifted))
                if shifted != current[new_key]:
                    relocated[new_key] = shifted
            if axis == "combined":
                for delay in delays_s if delays_s is not None else default_delays:
                    if values is not None:
                        if len(relocated) != 1:
                            raise ContractError("Combined absolute values require one control")
                        for value in values:
                            add(
                                {next(iter(relocated)): value}, delay, ["cell", "intensity", "time"]
                            )
                    else:
                        for plan in alternatives(
                            requested,
                            current,
                            "intensity",
                            {"changes": relocated, "delay_s": delay},
                            fractions=fractions,
                        ):
                            add(plan["changes"], delay, ["cell", "intensity", "time"])
            else:
                add(relocated, base.get("delay_s", 0), ["cell"])
    if axis not in ("all", "time", "intensity", "cell", "combined"):
        raise ContractError("Unknown exploration axis")
    seen, unique = set(), []
    for row in result:
        token = (tuple(sorted(row["changes"].items())), row["delay_s"])
        if token not in seen:
            seen.add(token)
            unique.append(row)
    return unique

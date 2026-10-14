"""Shared application contract; deployment roles do not change GNN inputs."""

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

REGISTRY = Path(__file__).parent / "context/app_registry_v16.json"


@dataclass(frozen=True)
class ControlApplication:
    app_id: str
    name: str
    role: str
    controls: tuple[str, ...]

    @property
    def target_ric(self):
        return "non_rt" if self.role == "rApp" else "near_rt"

    def describe(self):
        return dict(
            id=self.app_id,
            name=self.name,
            role=self.role,
            controls=list(self.controls),
            target_ric=self.target_ric,
        )

    def prepare(self, plan):
        """Validate ownership while preserving the exact reviewed intervention."""
        if plan.get("target_ric") != self.target_ric:
            raise ValueError(f"{self.name} requires the {self.target_ric} route")
        changes = plan["candidate"]["changes"]
        if not changes or not {key.split("_")[0] for key in changes} <= set(self.controls):
            raise ValueError(f"{self.name} cannot control the requested parameter family")
        if self.app_id == "energy_saving":
            if any(value >= plan["current_parameters"][key] for key, value in changes.items()):
                raise ValueError("Energy Saving rApp must reduce TxP")
        return deepcopy(plan)


def applications():
    return {
        row["id"]: ControlApplication(
            row["id"], row["label"] + " " + row["role"], row["role"], tuple(row["controls"])
        )
        for row in json.loads(REGISTRY.read_text(encoding="utf-8"))["apps"]
    }


def application_for(intent):
    return applications().get((intent or {}).get("objective"))


def prepare_application(plan, role=None):
    app = application_for(plan.get("intent"))
    if app is None:
        raise ValueError("Live deployment requires one of the five declared application intents")
    if role and app.role != role:
        raise ValueError(f"{app.name} must use its {app.target_ric} deployment route")
    return app, app.prepare(plan)

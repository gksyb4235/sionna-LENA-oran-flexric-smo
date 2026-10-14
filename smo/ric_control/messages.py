"""Wire messages shared by the Non-RT RIC and execution gateway."""

import hashlib
import json
from pydantic import BaseModel, ConfigDict, Field


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    live_run_id: str = Field(pattern="^[0-9a-f]{32}$")
    plan: dict


class RAppSubmission(Submission):
    policy_id: str = Field(pattern="^[0-9a-f]{32}$")
    policy_sha256: str = Field(pattern="^[0-9a-f]{64}$")


def submission_digest(body):
    value = body.model_dump(include={"live_run_id", "plan"})
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

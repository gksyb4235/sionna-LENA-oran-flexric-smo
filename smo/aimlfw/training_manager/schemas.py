"""HTTP request and response contracts for the Training Manager service."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from smo.aimlfw.common.constants import MAX_SEED
from smo.aimlfw.common.models import TrainingJob


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateJobRequest(ApiModel):
    feature_group: str = Field(min_length=1, max_length=128)
    model_name: str = Field(min_length=1, max_length=128)
    seed: int = Field(ge=0, le=MAX_SEED)
    metadata: dict[str, object] = Field(default_factory=dict)


class CreateJobResponse(ApiModel):
    job_id: str = Field(min_length=1)


class JobStatusResponse(TrainingJob):
    """The GET /jobs/{job_id} response is the TrainingJob snapshot itself."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

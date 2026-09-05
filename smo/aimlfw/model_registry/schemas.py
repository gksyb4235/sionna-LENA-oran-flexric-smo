"""HTTP contracts for the Model Registry service."""

from pydantic import BaseModel, ConfigDict, Field

from smo.aimlfw.common.models import ModelVersionRecord


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegisterModelVersionRequest(ApiModel):
    feature_group: str | None = None
    metrics: dict[str, float] | None = None
    artifact_uri: str | None = None
    train_split: float | None = None
    validation_split: float | None = None
    train_samples: int | None = None
    validation_samples: int | None = None
    seed: int | None = None


class RegisterModelVersionResponse(ApiModel):
    model_name: str = Field(min_length=1, max_length=128)
    version: int = Field(ge=1)


class ModelVersionListResponse(ApiModel):
    model_name: str = Field(min_length=1, max_length=128)
    versions: list[ModelVersionRecord] = Field(max_length=500)
    count: int = Field(ge=0, le=500)


class ServingVersionRequest(ApiModel):
    version: int = Field(ge=1)


class ServingVersionResponse(ApiModel):
    model_name: str = Field(min_length=1, max_length=128)
    version: int | None = Field(default=None, ge=1)

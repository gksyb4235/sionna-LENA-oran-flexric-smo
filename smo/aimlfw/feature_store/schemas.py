"""HTTP request and response contracts for the Feature Store."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from smo.aimlfw.common.models import FeatureRecord

MAX_RECORDS = 100_000


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UpsertRequest(ApiModel):
    records: list[FeatureRecord] = Field(max_length=MAX_RECORDS)


class UpsertResponse(ApiModel):
    upserted: int = Field(ge=0)
    total: int = Field(ge=0)


class SerializeRequest(ApiModel):
    feature_group: str = Field(min_length=1, max_length=128)
    records: list[FeatureRecord] = Field(max_length=MAX_RECORDS)
    target_path: str = Field(min_length=1)


class SerializeResponse(ApiModel):
    written: int = Field(ge=0)


class ParseRequest(ApiModel):
    source_path: str = Field(min_length=1)
    feature_group: str | None = Field(default=None, min_length=1, max_length=128)


class ParseResponse(ApiModel):
    records: list[FeatureRecord]
    count: int = Field(ge=0)


class RecordsResponse(ApiModel):
    records: list[FeatureRecord]
    count: int = Field(ge=0)

"""Atomic file-backed Feature Store and lossless CSV codec."""

from __future__ import annotations

import csv
import os
import tempfile
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from smo.aimlfw.common.config import ConfigurationError, FeatureGroupConfig, load_feature_group
from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES
from smo.aimlfw.common.models import FeatureRecord

from .errors import FeatureStoreError
from .schemas import MAX_RECORDS

_FIXED_PREFIX = ["feature_group", "time_step", "cell_id", *CONTROL_PARAMETER_NAMES]
_FIXED_SUFFIX = ["data_quality", "source_dir", "seed"]
_COUNT_SUFFIXES = ("__valid", "__excluded")


class FeatureStore:
    """Persist FeatureRecords by feature group using atomic CSV replacement."""

    def __init__(self, root: Path | str, config_dir: Path | str | None = None):
        self.root = Path(root)
        self.config_dir = Path(config_dir) if config_dir is not None else None
        self._thread_lock = threading.RLock()

    def _config(self, feature_group: str) -> FeatureGroupConfig:
        try:
            return load_feature_group(feature_group, self.config_dir)
        except ConfigurationError as exc:
            raise FeatureStoreError(
                "invalid_feature_group",
                "Feature Group configuration could not be loaded",
                {"feature_group": feature_group, "reason": str(exc)},
            ) from exc

    @staticmethod
    def _feature_names(config: FeatureGroupConfig) -> list[str]:
        return list(dict.fromkeys([*config.features, *config.target_kpis]))

    @classmethod
    def _header(cls, feature_names: Sequence[str]) -> list[str]:
        counts = [f"{name}{suffix}" for name in feature_names for suffix in _COUNT_SUFFIXES]
        return [*_FIXED_PREFIX, *feature_names, *counts, *_FIXED_SUFFIX]

    @staticmethod
    def _path_key(feature_group: str) -> str:
        # The config loader already restricts names to a safe filename alphabet.
        return f"{feature_group}.csv"

    def _store_path(self, feature_group: str) -> Path:
        self._config(feature_group)
        return self.root / self._path_key(feature_group)

    @contextmanager
    def _file_lock(self, feature_group: str, *, exclusive: bool) -> Iterator[None]:
        """Coordinate readers and writers across threads and local processes."""
        import fcntl

        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / f".{self._path_key(feature_group)}.lock"
        with self._thread_lock, lock_path.open("a", encoding="utf-8") as lock_file:
            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(lock_file.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _validate_records(
        self,
        feature_group: str,
        records: Sequence[FeatureRecord],
    ) -> tuple[FeatureGroupConfig, list[str]]:
        if len(records) > MAX_RECORDS:
            raise FeatureStoreError(
                "too_many_records",
                f"At most {MAX_RECORDS} Feature Records may be processed",
                {"received": len(records), "maximum": MAX_RECORDS},
            )
        config = self._config(feature_group)
        names = self._feature_names(config)
        expected = set(names)
        for index, record in enumerate(records):
            if record.feature_group != feature_group:
                raise FeatureStoreError(
                    "feature_group_mismatch",
                    "Every Feature Record must match the requested Feature Group",
                    {"record_index": index, "expected": feature_group, "actual": record.feature_group},
                )

            actual = set(record.features)
            if actual != expected:
                self._raise_schema_mismatch(expected, actual, record_index=index)
        return config, names

    @staticmethod
    def _raise_schema_mismatch(
        expected: set[str],
        actual: set[str],
        *,
        record_index: int | None = None,
    ) -> None:
        missing = sorted(expected - actual)
        undefined = sorted(actual - expected)
        details: dict[str, Any] = {
            "missing_fields": missing,
            "undefined_fields": undefined,
        }
        if record_index is not None:
            details["record_index"] = record_index
        code = "missing_fields" if missing else "undefined_fields"
        raise FeatureStoreError(code, "Feature Record field names do not match the Feature Group", details)

    @staticmethod
    def _record_row(record: FeatureRecord, feature_names: Sequence[str]) -> list[Any]:
        parameters = record.control_parameters
        row: list[Any] = [
            record.feature_group,
            record.time_step,
            record.cell_id,
            *(getattr(parameters, name) for name in CONTROL_PARAMETER_NAMES),
        ]
        row.extend("" if record.features[name] is None else record.features[name] for name in feature_names)
        for name in feature_names:
            count = record.sample_counts[name]
            row.extend((count.valid, count.excluded))
        row.extend((record.data_quality, record.source_dir, "" if record.seed is None else record.seed))
        return row

    @staticmethod
    def _sync_directory(directory: Path) -> None:
        try:
            descriptor = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def serialize(
        self,
        feature_group: str,
        records: Sequence[FeatureRecord],
        target_path: Path | str,
    ) -> int:
        """Sort, round-trip verify, and atomically replace one CSV file."""
        _, feature_names = self._validate_records(feature_group, records)
        ordered = sorted(records, key=lambda record: (record.time_step, record.cell_id))
        target = Path(target_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
            )
        except OSError as exc:
            raise FeatureStoreError(
                "file_write_failed",
                "Could not prepare the target file",
                {"target_path": str(target), "reason": str(exc)},
            ) from exc

        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
                writer = csv.writer(output, lineterminator="\n")
                writer.writerow(self._header(feature_names))
                writer.writerows(self._record_row(record, feature_names) for record in ordered)
                output.flush()
                os.fsync(output.fileno())
            self._verify_roundtrip(feature_group, ordered, temporary)
            os.replace(temporary, target)
            self._sync_directory(target.parent)
        except FeatureStoreError:
            temporary.unlink(missing_ok=True)
            raise
        except (OSError, UnicodeError, csv.Error) as exc:
            temporary.unlink(missing_ok=True)
            raise FeatureStoreError(
                "file_write_failed",
                "Feature Record serialization failed",
                {"target_path": str(target), "reason": str(exc)},
            ) from exc
        return len(ordered)

    def _verify_roundtrip(
        self,
        feature_group: str,
        expected: Sequence[FeatureRecord],
        source: Path,
    ) -> None:
        try:
            actual = self.parse(source, feature_group=feature_group)
        except FeatureStoreError as exc:
            details = {"cause": exc.error_code, **exc.details}
            details.setdefault("record_index", 0)
            details.setdefault("field_name", "unknown")
            raise FeatureStoreError(
                "roundtrip_violation",
                "Serialized Feature Records could not be parsed",
                details,
            ) from exc
        if len(actual) != len(expected):
            raise FeatureStoreError(
                "roundtrip_violation",
                "Serialized Feature Record count changed during parsing",
                {"record_index": min(len(actual), len(expected)), "field_name": "record_count"},
            )
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            mismatch = self._first_mismatch(left.model_dump(mode="json"), right.model_dump(mode="json"))
            if mismatch is not None:
                raise FeatureStoreError(
                    "roundtrip_violation",
                    "A Feature Record changed during serialization",
                    {"record_index": index, "field_name": mismatch},
                )

    @classmethod
    def _first_mismatch(cls, expected: Any, actual: Any, prefix: str = "") -> str | None:
        if isinstance(expected, dict) and isinstance(actual, dict):
            if set(expected) != set(actual):
                return prefix or "fields"
            for key in expected:
                mismatch = cls._first_mismatch(expected[key], actual[key], f"{prefix}.{key}".strip("."))
                if mismatch is not None:
                    return mismatch
            return None
        if isinstance(expected, list) and isinstance(actual, list):
            if len(expected) != len(actual):
                return prefix
            for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
                mismatch = cls._first_mismatch(left, right, f"{prefix}[{index}]")
                if mismatch is not None:
                    return mismatch
            return None
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return None if abs(float(expected) - float(actual)) <= 1e-9 else prefix
        return None if expected == actual else prefix

    def parse(self, source_path: Path | str, feature_group: str | None = None) -> list[FeatureRecord]:
        """Parse a complete CSV or fail without returning partial records."""
        source = Path(source_path)
        if not source.is_file():
            raise FeatureStoreError(
                "source_not_found",
                "Serialized Feature Record file does not exist",
                {"source_path": str(source)},
            )
        try:
            with source.open("r", encoding="utf-8", newline="") as input_file:
                reader = csv.reader(input_file, strict=True)
                header = next(reader, None)
                if header is None:
                    raise FeatureStoreError(
                        "missing_fields",
                        "Serialized Feature Record file has no header",
                        {"missing_fields": _FIXED_PREFIX + _FIXED_SUFFIX, "undefined_fields": []},
                    )
                rows = self._read_rows(reader, len(header))
        except FeatureStoreError:
            raise
        except (OSError, UnicodeError) as exc:
            raise FeatureStoreError(
                "file_read_failed",
                "Could not read serialized Feature Records",
                {"source_path": str(source), "reason": str(exc)},
            ) from exc
        except csv.Error as exc:
            raise FeatureStoreError(
                "truncated_record",
                "CSV syntax is corrupted",
                {"record_index": 0, "reason": str(exc)},
            ) from exc

        if len(rows) > MAX_RECORDS:
            raise FeatureStoreError(
                "too_many_records",
                f"At most {MAX_RECORDS} Feature Records may be parsed",
                {"received": len(rows), "maximum": MAX_RECORDS},
            )
        effective_group = feature_group or self._group_from_rows(header, rows)
        if effective_group is None:
            feature_names = self._feature_names_from_empty_header(header)
        else:
            feature_names = self._feature_names(self._config(effective_group))
        self._validate_header(header, feature_names)
        return [self._parse_row(header, row, feature_names, index, effective_group) for index, row in enumerate(rows)]

    @staticmethod
    def _read_rows(reader: Iterator[list[str]], field_count: int) -> list[list[str]]:
        rows: list[list[str]] = []
        while True:
            try:
                row = next(reader)
            except StopIteration:
                return rows
            except csv.Error as exc:
                raise FeatureStoreError(
                    "truncated_record",
                    "CSV record is truncated or malformed",
                    {"record_index": len(rows), "reason": str(exc)},
                ) from exc
            if len(row) != field_count:
                raise FeatureStoreError(
                    "truncated_record",
                    "CSV record field count does not match the header",
                    {
                        "record_index": len(rows),
                        "expected_field_count": field_count,
                        "actual_field_count": len(row),
                    },
                )
            rows.append(row)

    @staticmethod
    def _group_from_rows(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str | None:
        if not rows or "feature_group" not in header:
            return None
        return rows[0][header.index("feature_group")]

    @staticmethod
    def _feature_names_from_empty_header(header: Sequence[str]) -> list[str]:
        try:
            start = header.index(CONTROL_PARAMETER_NAMES[-1]) + 1
            end = min(header.index(field) for field in _FIXED_SUFFIX if field in header)
        except (ValueError, IndexError):
            start = len(_FIXED_PREFIX)
            end = len(header)
        candidates = header[start:end]
        return [name for name in candidates if not name.endswith(_COUNT_SUFFIXES)]

    @classmethod
    def _validate_header(cls, header: Sequence[str], feature_names: Sequence[str]) -> None:
        expected = cls._header(feature_names)
        duplicates = sorted({name for name in header if header.count(name) > 1})
        if duplicates:
            raise FeatureStoreError(
                "undefined_fields",
                "CSV header contains duplicate field names",
                {"missing_fields": [], "undefined_fields": duplicates},
            )
        if set(header) != set(expected):
            cls._raise_schema_mismatch(set(expected), set(header))
        if list(header) != expected:
            raise FeatureStoreError(
                "invalid_schema",
                "CSV header fields are not in canonical order",
                {"expected_fields": expected, "actual_fields": list(header)},
            )

    @staticmethod
    def _number(value: str, field_name: str, record_index: int, *, integer: bool = False) -> int | float:
        try:
            return int(value) if integer else float(value)
        except ValueError as exc:
            raise FeatureStoreError(
                "corrupted_record",
                "Feature Record contains an invalid numeric value",
                {"record_index": record_index, "field_name": field_name, "value": value},
            ) from exc

    @classmethod
    def _parse_row(
        cls,
        header: Sequence[str],
        row: Sequence[str],
        feature_names: Sequence[str],
        record_index: int,
        expected_group: str | None,
    ) -> FeatureRecord:
        values = dict(zip(header, row, strict=True))
        group = values["feature_group"]
        if expected_group is not None and group != expected_group:
            raise FeatureStoreError(
                "feature_group_mismatch",
                "Serialized file contains more than one Feature Group",
                {"record_index": record_index, "expected": expected_group, "actual": group},
            )
        features: dict[str, float | None] = {}
        counts: dict[str, dict[str, int]] = {}
        for name in feature_names:
            features[name] = None if values[name] == "" else float(
                cls._number(values[name], name, record_index)
            )
            counts[name] = {
                "valid": int(cls._number(values[f"{name}__valid"], f"{name}__valid", record_index, integer=True)),
                "excluded": int(
                    cls._number(values[f"{name}__excluded"], f"{name}__excluded", record_index, integer=True)
                ),
            }
        payload = {
            "feature_group": group,
            "time_step": cls._number(values["time_step"], "time_step", record_index, integer=True),
            "cell_id": values["cell_id"],
            "control_parameters": {
                "tx_power_dbm": cls._number(values["tx_power_dbm"], "tx_power_dbm", record_index),
                "ret_tilt_deg": cls._number(values["ret_tilt_deg"], "ret_tilt_deg", record_index),
                "cio_bias_db": cls._number(values["cio_bias_db"], "cio_bias_db", record_index),
                "hysteresis_db": cls._number(values["hysteresis_db"], "hysteresis_db", record_index),
                "ttt_ms": cls._number(values["ttt_ms"], "ttt_ms", record_index, integer=True),
            },
            "features": features,
            "sample_counts": counts,
            "data_quality": values["data_quality"],
            "source_dir": values["source_dir"],
            "seed": None
            if values["seed"] == ""
            else cls._number(values["seed"], "seed", record_index, integer=True),
        }
        try:
            return FeatureRecord.model_validate(payload)
        except ValidationError as exc:
            first = exc.errors(include_context=False)[0]
            field_name = ".".join(str(part) for part in first["loc"])
            raise FeatureStoreError(
                "corrupted_record",
                "Serialized Feature Record violates its schema",
                {
                    "record_index": record_index,
                    "field_name": field_name,
                    "reason": first["msg"],
                },
            ) from exc

    def upsert(self, records: Sequence[FeatureRecord]) -> tuple[int, int]:
        """Atomically insert or replace records by (feature_group, time_step, cell_id)."""
        if not records:
            return 0, 0
        groups = {record.feature_group for record in records}
        if len(groups) != 1:
            raise FeatureStoreError(
                "feature_group_mismatch",
                "One upsert request may contain exactly one Feature Group",
                {"feature_groups": sorted(groups)},
            )
        feature_group = next(iter(groups))
        self._validate_records(feature_group, records)
        path = self._store_path(feature_group)
        with self._file_lock(feature_group, exclusive=True):
            existing = self.parse(path, feature_group) if path.exists() else []
            indexed = {(record.time_step, record.cell_id): record for record in existing}
            incoming = {(record.time_step, record.cell_id): record for record in records}
            indexed.update(incoming)
            self.serialize(feature_group, list(indexed.values()), path)
        return len(incoming), len(indexed)

    def records(
        self,
        feature_group: str,
        *,
        cell_id: str | None = None,
        time_step: int | None = None,
    ) -> list[FeatureRecord]:
        """Read a stable snapshot, optionally filtered by key components."""
        path = self._store_path(feature_group)
        if not path.exists():
            return []
        with self._file_lock(feature_group, exclusive=False):
            records = self.parse(path, feature_group)
        return [
            record
            for record in records
            if (cell_id is None or record.cell_id == cell_id)
            and (time_step is None or record.time_step == time_step)
        ]

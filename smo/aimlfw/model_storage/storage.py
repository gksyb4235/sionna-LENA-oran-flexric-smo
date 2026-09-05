"""Local filesystem implementation of versioned model artifact storage."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

MODEL_FILENAME = "model.pt"


class ModelStorageError(Exception):
    """Base error raised by model storage operations."""


class InvalidArtifactIdentityError(ModelStorageError, ValueError):
    """Raised when a model name or version cannot form a safe storage key."""


class ArtifactAlreadyExistsError(ModelStorageError, FileExistsError):
    """Raised when an artifact already occupies a model/version key."""


ArtifactSource = bytes | bytearray | memoryview | str | os.PathLike[str] | BinaryIO


class ModelStorage:
    """Store one immutable artifact at ``models/<name>/<version>/model.pt``."""

    def __init__(self, root: str | os.PathLike[str] = "models") -> None:
        self.root = Path(root)

    @staticmethod
    def _validate_identity(model_name: str, version: int) -> None:
        if not isinstance(model_name, str) or not 1 <= len(model_name) <= 128:
            raise InvalidArtifactIdentityError("model_name must contain 1..128 characters")
        if model_name in {".", ".."} or not model_name.strip():
            raise InvalidArtifactIdentityError("model_name must be a non-empty path segment")
        if any(character in model_name for character in ("/", "\\", "\x00")):
            raise InvalidArtifactIdentityError("model_name must not contain path separators")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise InvalidArtifactIdentityError("version must be a positive integer")

    def artifact_path(self, model_name: str, version: int) -> Path:
        """Return the deterministic path for a model/version without creating it."""
        self._validate_identity(model_name, version)
        return self.root / model_name / str(version) / MODEL_FILENAME

    def path_for(self, model_name: str, version: int) -> Path:
        """Compatibility name for callers allocating the destination path."""
        return self.artifact_path(model_name, version)

    def save_artifact(self, model_name: str, version: int, source: ArtifactSource) -> str:
        """Atomically save an artifact and fail if the key already exists.

        The source may be bytes, a source file path, or a binary file-like object.
        The returned URI is the filesystem path accepted by ``artifact_exists``.
        """
        destination = self.artifact_path(model_name, version)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._require_inside_root(destination)
        temporary = destination.parent / f".{MODEL_FILENAME}.{uuid4().hex}.tmp"

        try:
            with temporary.open("xb") as output:
                self._copy_source(source, output)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise ArtifactAlreadyExistsError(
                    f"artifact already exists for {model_name!r} version {version}"
                ) from error
        finally:
            temporary.unlink(missing_ok=True)

        return destination.as_posix()

    def save(self, model_name: str, version: int, source: ArtifactSource) -> str:
        """Save an artifact; shorthand for :meth:`save_artifact`."""
        return self.save_artifact(model_name, version, source)

    def artifact_exists(self, uri: str | os.PathLike[str]) -> bool:
        """Return whether ``uri`` identifies a readable artifact in this storage root."""
        try:
            candidate = self._resolve_uri(uri)
            relative = candidate.relative_to(self.root.resolve())
            if len(relative.parts) != 3 or relative.parts[2] != MODEL_FILENAME:
                return False
            self._validate_identity(relative.parts[0], int(relative.parts[1]))
        except (InvalidArtifactIdentityError, TypeError, ValueError):
            return False
        return candidate.is_file()

    def _require_inside_root(self, path: Path) -> None:
        try:
            path.resolve().relative_to(self.root.resolve())
        except ValueError as error:
            raise InvalidArtifactIdentityError("artifact path escapes the storage root") from error

    def _resolve_uri(self, uri: str | os.PathLike[str]) -> Path:
        candidate = Path(uri)
        if not candidate.is_absolute():
            root_parts = self.root.parts
            if root_parts and candidate.parts[: len(root_parts)] == root_parts:
                pass
            elif candidate.parts and candidate.parts[0] == self.root.name:
                candidate = self.root.parent / candidate
            else:
                candidate = self.root / candidate
        resolved = candidate.resolve()
        resolved.relative_to(self.root.resolve())
        return resolved

    @staticmethod
    def _copy_source(source: ArtifactSource, output: BinaryIO) -> None:
        if isinstance(source, (bytes, bytearray, memoryview)):
            output.write(bytes(source))
            return
        if isinstance(source, (str, os.PathLike)):
            source_path = Path(source)
            if not source_path.is_file():
                raise FileNotFoundError(f"artifact source is not a file: {source_path}")
            with source_path.open("rb") as input_file:
                shutil.copyfileobj(input_file, output)
            return
        read = getattr(source, "read", None)
        if not callable(read):
            raise TypeError("source must be bytes, a file path, or a binary file-like object")
        shutil.copyfileobj(source, output)


def artifact_exists(
    uri: str | os.PathLike[str], storage_root: str | os.PathLike[str] = "models"
) -> bool:
    """Check an artifact URI against a Model Storage root."""
    return ModelStorage(storage_root).artifact_exists(uri)

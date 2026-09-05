from io import BytesIO
from pathlib import Path

import pytest

from smo.aimlfw.model_storage import (
    ArtifactAlreadyExistsError,
    InvalidArtifactIdentityError,
    ModelStorage,
    artifact_exists,
)


def test_save_assigns_unique_version_paths_and_preserves_bytes(tmp_path: Path) -> None:
    storage = ModelStorage(tmp_path / "models")

    first_uri = storage.save_artifact("ran-gnn", 1, b"version-one")
    second_uri = storage.save_artifact("ran-gnn", 2, b"version-two")

    first = Path(first_uri)
    second = Path(second_uri)
    assert first == tmp_path / "models" / "ran-gnn" / "1" / "model.pt"
    assert second == tmp_path / "models" / "ran-gnn" / "2" / "model.pt"
    assert first.read_bytes() == b"version-one"
    assert second.read_bytes() == b"version-two"
    assert storage.artifact_exists(first_uri)
    assert storage.artifact_exists(second_uri)


def test_save_never_overwrites_an_existing_version(tmp_path: Path) -> None:
    storage = ModelStorage(tmp_path / "models")
    uri = storage.save_artifact("ran-gnn", 7, b"original")

    with pytest.raises(ArtifactAlreadyExistsError):
        storage.save_artifact("ran-gnn", 7, b"replacement")

    assert Path(uri).read_bytes() == b"original"


def test_save_supports_source_files_and_binary_streams(tmp_path: Path) -> None:
    storage = ModelStorage(tmp_path / "models")
    source = tmp_path / "trained.pt"
    source.write_bytes(b"from-file")

    file_uri = storage.save_artifact("file-model", 1, source)
    stream_uri = storage.save_artifact("stream-model", 1, BytesIO(b"from-stream"))

    assert Path(file_uri).read_bytes() == b"from-file"
    assert Path(stream_uri).read_bytes() == b"from-stream"


def test_missing_source_leaves_no_partial_artifact(tmp_path: Path) -> None:
    storage = ModelStorage(tmp_path / "models")

    with pytest.raises(FileNotFoundError):
        storage.save_artifact("ran-gnn", 1, tmp_path / "missing.pt")

    assert not storage.artifact_path("ran-gnn", 1).exists()
    assert list((tmp_path / "models" / "ran-gnn" / "1").iterdir()) == []


def test_artifact_exists_only_accepts_canonical_storage_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "models"
    storage = ModelStorage(root)
    uri = storage.save_artifact("ran-gnn", 1, b"artifact")
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"not-managed")

    assert artifact_exists(uri, root)
    assert storage.artifact_exists("models/ran-gnn/1/model.pt")
    assert not storage.artifact_exists(outside)
    assert not storage.artifact_exists(root / "ran-gnn" / "1" / "other.pt")
    assert not storage.artifact_exists(root / "ran-gnn" / "2" / "model.pt")


@pytest.mark.parametrize(
    ("model_name", "version"),
    [
        ("", 1),
        ("   ", 1),
        ("../escape", 1),
        ("nested/model", 1),
        ("nested\\model", 1),
        ("ran-gnn", 0),
        ("ran-gnn", -1),
        ("ran-gnn", True),
        ("ran-gnn", 1.0),
    ],
)
def test_invalid_storage_keys_are_rejected(
    tmp_path: Path, model_name: str, version: object
) -> None:
    storage = ModelStorage(tmp_path / "models")

    with pytest.raises(InvalidArtifactIdentityError):
        storage.artifact_path(model_name, version)  # type: ignore[arg-type]

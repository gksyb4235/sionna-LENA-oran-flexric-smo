"""Property 26 test module (own module to avoid collisions with parallel Property tasks).

Task 3.15: 임의의 (모델 이름, 버전 번호) 조합 집합에 대해 Model_Storage가 할당하는
산출물 경로가 모두 서로 다르고, 새 버전 등록이 기존 버전의 산출물 파일을 덮어쓰거나
삭제하지 않는지를 검증한다.

Targets ``smo.aimlfw.model_storage.storage.ModelStorage`` (Task 3.1), as specified in
requirements.md Requirement 4.6:

    THE Model_Storage SHALL 등록된 각 (모델 이름, 버전 번호) 조합의 산출물 파일을
    서로 겹치지 않는 고유 위치에 보관하고, 다른 버전의 산출물 파일을 덮어쓰거나
    삭제하지 않는다.

design.md Property 26: 산출물 경로 고유성
    *For any* 서로 다른 (모델 이름, 버전 번호) 조합 집합, Model_Storage가 할당하는
    산출물 경로는 모두 서로 다르며, 새 버전 등록이 기존 버전의 산출물 파일을
    덮어쓰거나 삭제하지 않는다.
    **Validates: Requirements 4.6**
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.model_storage import ArtifactAlreadyExistsError, ModelStorage
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS

# Model names limited to characters accepted by ``ModelStorage._validate_identity``
# (no path separators or NUL bytes) so every drawn name is a legal storage key.
_MODEL_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-",
    min_size=1,
    max_size=32,
).filter(lambda name: name.strip() and name not in {".", ".."})
_VERSION = st.integers(min_value=1, max_value=1_000)


@st.composite
def _distinct_model_version_pairs(draw: Any) -> list[tuple[str, int]]:
    """Draw a set of distinct (model_name, version) identity pairs."""
    pairs = draw(
        st.lists(st.tuples(_MODEL_NAME, _VERSION), min_size=1, max_size=15, unique=True)
    )
    return pairs


# **Property 26: 산출물 경로 고유성**
# **Validates: Requirements 4.6**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(pairs=_distinct_model_version_pairs())
def test_artifact_paths_are_unique_and_non_overwriting(pairs: list[tuple[str, int]]) -> None:
    # A fresh temporary storage root per generated example: ``tmp_path`` is a
    # function-scoped pytest fixture and is not reset between Hypothesis examples,
    # so it cannot be used as a @given parameter here.
    with tempfile.TemporaryDirectory() as raw_root:
        storage = ModelStorage(Path(raw_root) / "models")

        # 1. Distinct (model_name, version) identities must be allocated distinct paths.
        paths = [storage.artifact_path(model_name, version) for model_name, version in pairs]
        assert len(set(paths)) == len(pairs)

        # 2. Saving an artifact for every pair must never disturb a previously saved
        # artifact belonging to a different (model_name, version) key.
        contents = {
            (model_name, version): f"{model_name}:{version}".encode()
            for model_name, version in pairs
        }
        for model_name, version in pairs:
            storage.save_artifact(model_name, version, contents[(model_name, version)])

        for model_name, version in pairs:
            stored_path = storage.artifact_path(model_name, version)
            assert stored_path.read_bytes() == contents[(model_name, version)]

        # 3. Re-registering any existing (model_name, version) key is rejected and the
        # original artifact's bytes remain unchanged (non-overwrite guarantee).
        sample_name, sample_version = pairs[0]
        with pytest.raises(ArtifactAlreadyExistsError):
            storage.save_artifact(sample_name, sample_version, b"attempted-overwrite")

        assert (
            storage.artifact_path(sample_name, sample_version).read_bytes()
            == contents[(sample_name, sample_version)]
        )

"""Unit tests for qml_observer.core.run."""

import pytest

from qml_observer.core.run import DEFAULT_RUN_ID_PREFIX, generate_run_id, validate_run_id


class TestGenerateRunId:
    def test_default_prefix(self):
        run_id = generate_run_id()
        assert run_id.startswith(f"{DEFAULT_RUN_ID_PREFIX}-")

    def test_custom_prefix(self):
        run_id = generate_run_id(prefix="vqe")
        assert run_id.startswith("vqe-")

    def test_format_has_12_hex_chars(self):
        run_id = generate_run_id()
        suffix = run_id.split("-", 1)[1]
        assert len(suffix) == 12
        int(suffix, 16)  # raises if not valid hex

    def test_uniqueness(self):
        ids = {generate_run_id() for _ in range(1000)}
        assert len(ids) == 1000

    def test_empty_prefix_raises(self):
        with pytest.raises(ValueError):
            generate_run_id(prefix="")

    def test_non_str_prefix_raises(self):
        with pytest.raises(TypeError):
            generate_run_id(prefix=123)


class TestValidateRunId:
    def test_valid_passthrough(self):
        assert validate_run_id("my-run") == "my-run"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            validate_run_id("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            validate_run_id("   ")

    def test_non_str_raises(self):
        with pytest.raises(TypeError):
            validate_run_id(123)

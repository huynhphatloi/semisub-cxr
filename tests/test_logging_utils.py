"""Unit tests for src/utils/logging_utils.py."""

import logging
import os
import sys

import numpy as np
import torch

from src.utils.logging_utils import get_git_commit, log_library_versions, setup_logging


def test_setup_logging_returns_logger(tmp_path):
    """setup_logging returns a logging.Logger with the experiment name."""
    logger = setup_logging(str(tmp_path), "test_exp")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_exp"


def test_setup_logging_creates_log_file(tmp_path):
    """setup_logging creates a .log file in the output directory."""
    setup_logging(str(tmp_path), "my_experiment")
    log_file = tmp_path / "my_experiment.log"
    assert log_file.exists()


def test_setup_logging_creates_output_dir(tmp_path):
    """setup_logging creates the output directory if it does not exist."""
    new_dir = tmp_path / "nested" / "logs"
    setup_logging(str(new_dir), "exp")
    assert new_dir.exists()


def test_setup_logging_writes_to_file(tmp_path):
    """Messages logged at INFO level appear in the log file."""
    logger = setup_logging(str(tmp_path), "write_test")
    logger.info("hello from test")
    # Flush handlers
    for handler in logger.handlers:
        handler.flush()
    log_content = (tmp_path / "write_test.log").read_text()
    assert "hello from test" in log_content


def test_setup_logging_no_duplicate_handlers(tmp_path):
    """Calling setup_logging twice does not add duplicate handlers."""
    logger1 = setup_logging(str(tmp_path), "dup_test")
    n_handlers = len(logger1.handlers)
    logger2 = setup_logging(str(tmp_path), "dup_test")
    assert logger1 is logger2
    assert len(logger2.handlers) == n_handlers


def test_get_git_commit_returns_string_or_none():
    """get_git_commit returns a non-empty string (in a git repo) or None."""
    result = get_git_commit()
    if result is not None:
        assert isinstance(result, str)
        assert len(result) > 0


def test_log_library_versions_keys():
    """log_library_versions returns dict with torch, numpy, python keys."""
    versions = log_library_versions()
    assert set(versions.keys()) == {"torch", "numpy", "python"}


def test_log_library_versions_values():
    """log_library_versions returns correct version strings."""
    versions = log_library_versions()
    assert versions["torch"] == torch.__version__
    assert versions["numpy"] == np.__version__
    assert versions["python"] == sys.version.split()[0]

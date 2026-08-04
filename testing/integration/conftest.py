"""Pytest configuration for Phase 1 and Phase 2 real-data integration tests."""
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: marks tests as real-data integration tests")
    config.addinivalue_line("markers", "gpu: marks tests as requiring a CUDA GPU")
    config.addinivalue_line("markers", "slow: marks tests as slow (smoke runs, etc.)")
    config.addinivalue_line("markers", "depth: marks tests as Phase 2 depth-anchor integration tests")
    config.addinivalue_line("markers", "extended: marks tests as part of the opt-in extended matrix")


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def is_integration_enabled():
    return os.environ.get("AI_TOOLKIT_RUN_KREA_INTEGRATION", "0") == "1"


@pytest.fixture(scope="session")
def has_cuda():
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def dataset_root():
    explicit = os.environ.get("AI_TOOLKIT_TEST_DATASET")
    if explicit and os.path.isdir(explicit):
        return explicit
    default = os.path.join(REPO_ROOT, "datasets")
    if os.path.isdir(default):
        return default
    return None


@pytest.fixture(scope="session")
def output_root():
    explicit = os.environ.get("AI_TOOLKIT_TEST_OUTPUT")
    if explicit:
        return explicit
    return os.path.join(REPO_ROOT, "test_outputs", "phase1_real_data")


# ---------------------------------------------------------------------------
# Phase 2 (depth-anchor) fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def is_depth_integration_enabled():
    """Phase 2 tests require an explicit AI_TOOLKIT_RUN_KREA_DEPTH_INTEGRATION=1."""
    return os.environ.get("AI_TOOLKIT_RUN_KREA_DEPTH_INTEGRATION", "0") == "1"


@pytest.fixture(scope="session")
def depth_dataset_root():
    """Dataset root for depth tests.

    Resolution order: AI_TOOLKIT_TEST_DATASET, then the repository ``datasets``
    directory, then None (skip). Same source as Phase 1 -- the real images live
    under ``datasets/Sample_{512,768,1024}``.
    """
    explicit = os.environ.get("AI_TOOLKIT_TEST_DATASET")
    if explicit and os.path.isdir(explicit):
        return explicit
    default = os.path.join(REPO_ROOT, "datasets")
    if os.path.isdir(default):
        return default
    return None


@pytest.fixture(scope="session")
def depth_output_root():
    explicit = os.environ.get("AI_TOOLKIT_DEPTH_TEST_OUTPUT")
    if explicit:
        return explicit
    return os.path.join(REPO_ROOT, "test_outputs", "phase2_real_data")

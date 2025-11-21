"""Pytest configuration and fixtures."""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires models)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (may take minutes)"
    )


@pytest.fixture
def sample_attribution_scores():
    """Provide sample attribution scores for testing."""
    import numpy as np
    np.random.seed(42)
    # 20 inferences, 8 layers, 16 nodes per layer
    return np.random.rand(20, 8, 16)


@pytest.fixture
def sample_positions():
    """Provide sample node positions for testing."""
    # 8 layers, select 4 nodes per layer
    return [[0, 4, 8, 12] for _ in range(8)]

"""
Pytest configuration and fixtures
"""

import pytest

@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'model_name': 'pinn',
        'patient': 3,
        'mode': 'inverse',
    }

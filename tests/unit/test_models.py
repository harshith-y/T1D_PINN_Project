"""
Unit tests for model architectures
"""

import pytest
import tensorflow as tf


def test_tensorflow_import():
    """Test TensorFlow import"""
    assert tf.__version__ is not None


def test_tensorflow_version():
    """Test TensorFlow version is 2.x"""
    major_version = int(tf.__version__.split(".")[0])
    assert major_version >= 2, f"Expected TensorFlow 2.x, got {tf.__version__}"


# TODO: Add actual model tests

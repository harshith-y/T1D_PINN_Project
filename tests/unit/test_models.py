"""
Unit tests for model architectures
"""

import pytest
import tensorflow as tf
import torch


def test_tensorflow_import():
    """Test TensorFlow import"""
    assert tf.__version__ is not None


def test_pytorch_import():
    """Test PyTorch import"""
    assert torch.__version__ is not None


# TODO: Add actual model tests

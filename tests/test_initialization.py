import pytest
from llamea import LLaMEA


def f(ind, logger):
    return f"feedback {ind.name}", 1.0, ""


def test_default_initialization():
    """Test the default initialization of the LLaMEA class."""
    optimizer = LLaMEA(f, api_key="test_key")
    assert optimizer.api_key == "test_key"
    assert optimizer.model == "gpt-4-turbo", "Default model should be gpt-4-turbo"


def test_custom_initialization():
    """Test custom initialization parameters."""
    optimizer = LLaMEA(f, api_key="test_key", model="custom-model", budget=500)
    assert optimizer.model == "custom-model"
    assert optimizer.budget == 500, "Custom budget should be respected"

import pytest
from unittest.mock import MagicMock
from llamea import LLaMEA



def test_algorithm_generation():
    """Test the algorithm generation process."""

    def f(code, name, longname):
        return f"feedback {name}", 1.0, ""

    optimizer = LLaMEA(f, api_key="test_key", experiment_name="test generation")
    optimizer.llm = MagicMock(return_value=("class ExampleAlgorithm(object):\n    pass", "ExampleAlgorithm", "Long Example Algorithm"))

    algorithm_code, algorithm_name, algorithm_long_name = optimizer.llm(session_messages=[{"role": "system", "content": "test prompt"}])

    
    assert algorithm_long_name == "Long Example Algorithm", "Algorithm long name should be extracted correctly"
    assert algorithm_name == "ExampleAlgorithm", "Algorithm name should be extracted correctly"
    assert "class ExampleAlgorithm" in algorithm_code, "Algorithm code should be extracted correctly"

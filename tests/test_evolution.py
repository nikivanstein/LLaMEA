import pytest
from unittest.mock import MagicMock
from llamea import LLaMEA


def test_evolutionary_process():
    """Test the evolutionary process loop to ensure it updates generations."""

    def f(code, name, longname):
        return f"feedback {name}", 1.0, ""

    optimizer = LLaMEA(f, api_key="test_key", experiment_name="test evolution", budget=2)
    optimizer.llm = MagicMock(return_value=("class ExampleAlgorithm(object):\n    pass", "ExampleAlgorithm", "Long Example Algorithm"))
    best_solution, best_fitness = optimizer.run()  # Assuming run has a very simple loop for testing
    assert best_solution == "class ExampleAlgorithm(object):\n    pass", "best should be class ExampleAlgorithm(object):\n    pass"
    assert best_fitness == 1.0, "Fitness should be 1.0"
    assert optimizer.generation == 2, "Generation should increment correctly"

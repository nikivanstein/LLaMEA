import pytest
from unittest.mock import MagicMock
from llamea import LLaMEA

#Helper
class obj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, obj(v) if isinstance(v, dict) else v)

def test_evolutionary_process():
    """Test the evolutionary process loop to ensure it updates generations."""

    def f(code, name, longname):
        return f"feedback {name}", 1.0, ""

    
    response = obj({
        "choices": [
            obj({
                "index": 0,
                "finish_reason": "stop",
                "message": {"content": "# Name: Long Example Algorithm\n# Code:\n```python\nclass ExampleAlgorithm:\n    pass\n```", "role": "assistant"},
            })
        ]
    })
    
    optimizer = LLaMEA(f, api_key="test_key", experiment_name="test evolution", budget=2, log=False)
    optimizer.client.chat = MagicMock(return_value=response)
    best_solution, best_fitness = optimizer.run()  # Assuming run has a very simple loop for testing
    assert best_solution == "class ExampleAlgorithm:\n    pass", "best should be class ExampleAlgorithm(object):\n    pass"
    assert best_fitness == 1.0, "Fitness should be 1.0"
    assert optimizer.generation == 2, "Generation should increment correctly"

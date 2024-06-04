import pytest
from unittest.mock import MagicMock
from llamea import LLaMEA
import numpy as np

#Helper
class obj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, obj(v) if isinstance(v, dict) else v)

f_try = 0
def f(code, name, longname):
    global f_try
    f_try += 1
    return f"feedback {name} {f_try}", f_try*0.1, ""

def test_evolutionary_process():
    """Test the evolutionary process loop to ensure it updates generations."""
    
    response = obj({
        "choices": [
            obj({
                "index": 0,
                "finish_reason": "stop",
                "message": {"content": "# Name: Long Example Algorithm\n# Code:\n```python\nclass ExampleAlgorithm:\n    pass\n```", "role": "assistant"},
            })
        ]
    })
    
    optimizer = LLaMEA(f, api_key="test_key", experiment_name="test evolution", elitism=True, budget=10, log=True)
    optimizer.client.chat = MagicMock(return_value=response)
    best_solution, best_fitness = optimizer.run()  # Assuming run has a very simple loop for testing
    assert best_solution == "class ExampleAlgorithm:\n    pass", "best should be class ExampleAlgorithm(object):\n    pass"
    assert best_fitness == 1.0, "Fitness should be 1.0"
    assert optimizer.generation == 10, "Generation should increment correctly"

def test_evolutionary_process_with_errors():
    """Test the evolutionary process loop to ensure it updates generations even when errors occur."""
    
    response = obj({
        "choices": [
            obj({
                "index": 0,
                "finish_reason": "stop",
                "message": {"content": "hi!", "role": "assistant"},
            })
        ]
    })
    
    optimizer = LLaMEA(f, role_prompt="You are super cute", task_prompt="just say hi", api_key="test_key", experiment_name="test evolution with errors", budget=10, log=False)
    optimizer.client.chat = MagicMock(return_value=response)
    best_solution, best_fitness = optimizer.run()  # Assuming run has a very simple loop for testing
    assert best_fitness ==-np.Inf, f"Fitness should be 0.0 is {best_fitness}"
    assert optimizer.generation == 10, "Generation should increment correctly"
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

def test_algorithm_generation():
    """Test the algorithm generation process."""

    def f(code, name, longname):
        return f"feedback {name}", 1.0, ""

    optimizer = LLaMEA(f, api_key="test_key", experiment_name="test generation", log=False)
    response = obj({
        "choices": [
            obj({
                "index": 0,
                "finish_reason": "stop",
                "message": {"content": "# Name: Long Example Algorithm\n# Code:\n```python\nclass ExampleAlgorithm:\n    pass\n```", "role": "assistant"},
            })
        ]
    })
    optimizer.client.chat = MagicMock(return_value=response)

    algorithm_code, algorithm_name, algorithm_long_name = optimizer.llm(session_messages=[{"role": "system", "content": "test prompt"}])

    
    assert algorithm_long_name == "Long Example Algorithm", f"Algorithm long name should be extracted correctly, is {algorithm_long_name}"
    assert algorithm_name == "ExampleAlgorithm", "Algorithm name should be extracted correctly"
    assert "class ExampleAlgorithm" in algorithm_code, "Algorithm code should be extracted correctly"

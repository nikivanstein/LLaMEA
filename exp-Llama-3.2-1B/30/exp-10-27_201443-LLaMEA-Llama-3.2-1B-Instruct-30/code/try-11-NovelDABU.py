import numpy as np
from scipy.optimize import differential_evolution

class NovelDABU:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        # Refine the search space using probability 0.3
        probability = np.random.rand(2)  # 0.3 chance to refine the search space
        if probability < 0.3:
            self.search_space = np.linspace(0.0, 10.0, self.dim)  # refine the lower bound
        else:
            self.search_space = np.linspace(-5.0, 5.0, self.dim)  # refine the upper bound
        # Optimize the function using differential evolution
        res = differential_evolution(func, self.search_space)
        # Update the function evaluations
        self.func_evaluations += 1
        return res.fun

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

novel_dabu = NovelDABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(novel_dabu(test_function))  # prints a random value between -10 and 10

# Test the function
def test_function_refined(x):
    return np.exp(-x[0]**2 - x[1]**2)

refined_dabu = NovelDABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(refined_dabu(test_function_refined))  # prints a random value between -10 and 10
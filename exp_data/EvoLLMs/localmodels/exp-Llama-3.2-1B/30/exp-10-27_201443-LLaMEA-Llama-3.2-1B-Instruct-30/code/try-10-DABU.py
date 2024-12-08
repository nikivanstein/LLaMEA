import numpy as np
import random
from collections import deque

class DABU:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = deque(maxlen=self.budget)

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
            self.population.append(func_value)
            if len(self.population) > self.budget:
                random.shuffle(self.population)
        return random.choice(self.population)

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

dabu = DABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(dabu(test_function))  # prints a random value between -10 and 10

# Novel Metaheuristic Algorithm: Adaptive Multi-Step Greedy (AMSG)
# Description: Adaptive Multi-Step Greedy algorithm that adapts its strategy based on the performance of its solutions.
# Code: 
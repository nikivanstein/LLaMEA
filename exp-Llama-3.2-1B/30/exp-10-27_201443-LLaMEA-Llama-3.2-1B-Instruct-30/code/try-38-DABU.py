import numpy as np
import random
import matplotlib.pyplot as plt

class DABU:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.fitness_history = [0.01] * budget

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def select_strategy(self, func, budget):
        if budget > self.budget:
            return self.budget
        else:
            if random.random() < 0.3:
                return 1000
            else:
                return 500

    def mutate(self, func, mutation_rate):
        for _ in range(self.select_strategy(func, self.budget)):
            func_value = func(self.search_space)
            if np.random.rand() < mutation_rate:
                self.search_space = self.search_space + np.random.uniform(-1, 1, self.dim)

    def __repr__(self):
        return f"DABU(budget={self.budget}, dim={self.dim})"

# Description: DABU algorithm with adaptive search space and mutation strategy.
# Code: 
# ```python
# DABU(budget=1000, dim=2)
# DABU(budget=1000, dim=2)
# ```
dabu = DABU(1000, 2)  # 1000 function evaluations, 2 dimensions
dabu.mutate(test_function, 0.01)  # mutate the search space

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

# Plot the fitness curve
plt.plot(dabu.fitness_history)
plt.xlabel('Evaluation')
plt.ylabel('Fitness')
plt.title('Fitness over Convergence Curve')
plt.show()
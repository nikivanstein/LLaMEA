# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
from scipy.optimize import minimize
import random
import copy

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

    def __str__(self):
        return f"Black Box Optimizer (BBOB) with dimensionality {self.dim}, budget {self.budget}"

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Novel Metaheuristic Algorithm: "Refine and Adapt"
# Description: A novel metaheuristic algorithm that refines its strategy based on the selected solution and adapts to the search space.
# Code: 
# ```python
class RefineAndAdaptOptimizer(BBOBBlackBoxOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.search_space_copy = copy.deepcopy(self.search_space)
        self.best_individual = None
        self.best_fitness = float('inf')
        self.refine_strategy = False

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
            self.search_space_copy = copy.deepcopy(self.search_space)
            self.best_individual = result.x
            self.best_fitness = result.fun
            if self.best_fitness < self.best_fitness(self.best_individual):
                self.refine_strategy = True
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

    def refine_strategy(self):
        if self.refine_strategy:
            new_individual = copy.deepcopy(self.best_individual)
            new_individual = random.uniform(self.search_space[0], self.search_space[1])
            self.search_space[0] = new_individual
            self.search_space[1] = new_individual
            self.best_individual = new_individual
            self.best_fitness = self.evaluate_fitness(self.best_individual)
            print(f"Refined strategy: {self.best_individual}")
            print(f"Refined fitness: {self.best_fitness}")

    def evaluate_fitness(self, individual):
        return self.func(individual)

# Example usage:
optimizer = RefineAndAdaptOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Update the selected solution using the refine_strategy method
optimizer.refine_strategy()
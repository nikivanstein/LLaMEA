# Description: Novel metaheuristic algorithm for solving black box optimization problems
# Code: 
import numpy as np
from scipy.optimize import minimize

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0
        self.search_space_bounds = [(x, x) for x in self.search_space]
        self.search_space_bounds_dict = dict(zip(self.search_space_bounds, self.search_space_bounds))
        self.new_individual = None

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=self.search_space_bounds_dict)
            self.new_individual = result.x
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Novel metaheuristic algorithm: Evolutionary Multi-Objective Optimization (EMO)
# Description: EMO is a novel metaheuristic algorithm for solving multi-objective optimization problems
# Code: 
# ```python
# import numpy as np
# import random
# from scipy.optimize import minimize

class EMO(BBOBBlackBoxOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.population_size = 100
        self.population = [random.uniform(self.search_space_bounds_dict[i], self.search_space_bounds_dict[i+1]) for i in range(self.dim)]
        self.fitness_scores = [0] * self.population_size
        self.mutation_rate = 0.01
        self.crossover_rate = 0.5

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=self.search_space_bounds_dict)
            self.new_individual = result.x
            self.fitness_scores = [func(individual) for individual in self.population]
            self.fitness_scores.sort(key=lambda x: x, reverse=True)
            self.population = self.population[:self.population_size // 2]
            self.population.extend(self.population[self.population_size // 2:])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

# Example usage:
emo = EMO(1000, 10)
func = lambda x: x**2
result = emo(func)
print(result)
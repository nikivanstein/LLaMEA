# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            if func_value > 0.9:
                # Refine the strategy by increasing the mutation rate
                self.search_space = np.linspace(-5.0, 5.0, self.dim) + np.random.uniform(-0.1, 0.1, self.dim)
            elif func_value < 0.1:
                # Refine the strategy by decreasing the mutation rate
                self.search_space = np.linspace(-5.0, 5.0, self.dim) - np.random.uniform(-0.1, 0.1, self.dim)
            self.func_evaluations = 0
        return func_value

class MutationExp:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func):
        while True:
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            if np.random.rand() < 0.05:
                # Randomly select a mutation point
                mutation_point = np.random.choice(self.search_space.shape[0])
                # Mutate the individual at the selected point
                self.search_space[mutation_point] += np.random.uniform(-0.1, 0.1)
            self.func_evaluations += 1
            if self.func_evaluations >= self.budget:
                break
        return func_value

class HBEBOv2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            if np.random.rand() < 0.05:
                # Randomly select a mutation point
                mutation_point = np.random.choice(self.search_space.shape[0])
                # Mutate the individual at the selected point
                self.search_space[mutation_point] += np.random.uniform(-0.1, 0.1)
            elif np.random.rand() < 0.05:
                # Randomly select a crossover point
                crossover_point = np.random.choice(self.search_space.shape[0])
                # Crossover the two parent individuals
                self.search_space = np.concatenate((self.search_space[:crossover_point], self.search_space[crossover_point+1:]))
            self.func_evaluations = 0
        return func_value

# Test the algorithms
hebbbo = HEBBO(100, 10)
hebbbo_func = np.linspace(-5.0, 5.0, 10)
hebbbo_func_evaluations = 0
hebbbo_func_value = hebbbo(func_evaluations=hebbbo_func)
print(f"Hebbbo: {hebbbo_func_evaluations} evaluations, {hebbbo_func_value}")

mutation_exp = MutationExp(100, 10)
mutation_exp_func = np.linspace(-5.0, 5.0, 10)
mutation_exp_func_evaluations = 0
mutation_exp_func_value = mutation_exp(func_evaluations=mutation_exp_func)
print(f"Mutation Exp: {mutation_exp_func_evaluations} evaluations, {mutation_exp_func_value}")

hebbbov2 = HBEBOv2(100, 10)
hebbbov2_func = np.linspace(-5.0, 5.0, 10)
hebbbov2_func_evaluations = 0
hebbbov2_func_value = hebbbov2(func_evaluations=hebbbov2_func)
print(f"Hebbbo v2: {hebbbov2_func_evaluations} evaluations, {hebbbov2_func_value}")
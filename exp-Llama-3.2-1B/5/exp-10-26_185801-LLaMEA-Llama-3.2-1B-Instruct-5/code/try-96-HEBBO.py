import numpy as np
from scipy.optimize import differential_evolution

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
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def mutate(self, individual):
        # Select the individual with the highest fitness value
        max_individual = np.argmax([self.func(individual) for ind in self.func_evaluations:])
        
        # Generate two new individuals with the same fitness value as the selected individual
        new_individual1 = individual[:self.dim//2] + [self.search_space[self.dim//2]] + individual[self.dim//2+1:]
        new_individual2 = individual[:self.dim//2] + [self.search_space[0]] + individual[self.dim//2+1:]
        
        # Refine the selected individual's strategy by changing the lower and upper bounds of its search space
        if np.random.rand() < 0.5:
            self.search_space[:self.dim//2] = [self.search_space[self.dim//2] + 0.1]
            self.search_space[self.dim//2+1:] = [self.search_space[self.dim//2+1] - 0.1]
        else:
            self.search_space[:self.dim//2] = [self.search_space[self.dim//2] - 0.1]
            self.search_space[self.dim//2+1:] = [self.search_space[self.dim//2+1] + 0.1]
        
        return new_individual1, new_individual2

# One-line description: A novel metaheuristic algorithm that uses differential evolution to optimize black box functions with a wide range of tasks.
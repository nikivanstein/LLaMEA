import numpy as np
from scipy.optimize import minimize

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

    def adaptive_mutation(self, individual, mutation_rate):
        if np.random.rand() < mutation_rate:
            individual = np.random.uniform(self.search_space[0], self.search_space[1])
        return individual

    def adaptive_evolution_strategy(self, individual, mutation_rate):
        for _ in range(self.budget):
            new_individual = self.adaptive_mutation(individual, mutation_rate)
            result = self(__call__, new_individual)
            if result is not None:
                return result
        return None
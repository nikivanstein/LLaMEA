import numpy as np
import random
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

    def mutate(self, individual):
        if random.random() < 0.2:
            return individual + random.uniform(-0.5, 0.5)
        else:
            return individual

    def __call__(self, func):
        best_individual = None
        best_fitness = float('-inf')

        for _ in range(self.budget):
            individual = random.uniform(self.search_space)
            fitness = func(individual)
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual

            # Refine the strategy
            if random.random() < 0.2:
                best_individual = self.mutate(best_individual)

            # Select the best individual
            if random.random() < 0.5:
                individual = best_individual
            else:
                individual = random.choice(self.search_space)

        # Evaluate the best individual
        best_individual = self.evaluate_fitness(best_individual)

        return best_individual, best_fitness

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)
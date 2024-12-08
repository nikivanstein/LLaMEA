import numpy as np
from scipy.optimize import minimize
import random

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0
        self.population_size = 100

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

    def select_solution(self):
        # Refine the strategy based on the fitness and the number of evaluations
        fitnesses = []
        for _ in range(self.population_size):
            individual = random.uniform(self.search_space)
            fitness = func(individual)
            fitnesses.append(fitness)

        # Select the individual with the highest fitness
        selected_individual = np.argmax(fitnesses)

        # Change the strategy to refine the search space
        if self.dim > 2:
            # Select a new individual with a better fitness
            new_individual = random.uniform(self.search_space)
            fitness = func(new_individual)
            if fitness > self.search_space[selected_individual]:
                selected_individual = new_individual

        # Change the strategy to avoid getting stuck in local optima
        if np.random.rand() < 0.2:
            # Select a new individual with a worse fitness
            new_individual = random.uniform(self.search_space)
            fitness = func(new_individual)
            if fitness < self.search_space[selected_individual]:
                selected_individual = new_individual

        return selected_individual

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Print the updated population
print("Updated Population:")
for i, individual in enumerate(optimizer.population):
    print(f"Individual {i+1}: {individual}, Fitness: {func(individual)}")
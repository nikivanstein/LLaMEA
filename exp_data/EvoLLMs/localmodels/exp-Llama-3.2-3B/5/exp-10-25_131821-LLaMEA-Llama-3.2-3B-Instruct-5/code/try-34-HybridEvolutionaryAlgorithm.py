import numpy as np
from scipy.optimize import differential_evolution

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.probability = 0.05

    def __call__(self, func):
        # Initialize population with random points
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget, self.dim))

        # Initialize elite set
        elite_set = population[:int(self.budget * self.elitism_ratio)]

        # Perform probabilistic selection
        selected_population = elite_set + np.random.choice(population, size=self.budget - len(elite_set), p=np.ones(len(elite_set)) * self.elitism_ratio + (1 - self.elitism_ratio) * self.probability)

        # Perform differential evolution
        for _ in range(self.budget - len(selected_population)):
            # Evaluate population
            fitness = np.array([func(x) for x in selected_population])

            # Perform differential evolution
            new_population = differential_evolution(func, self.search_space, x0=selected_population, popsize=len(selected_population) + 1, maxiter=1)

            # Update population
            selected_population = np.concatenate((selected_population, new_population[0:1]))

        # Return the best solution
        return np.min(func(selected_population))

# Example usage:
def func(x):
    return np.sum(x**2)

hybrid_evolutionary_algorithm = HybridEvolutionaryAlgorithm(budget=100, dim=10)
best_solution = hybrid_evolutionary_algorithm(func)
print(f"Best solution: {best_solution}")
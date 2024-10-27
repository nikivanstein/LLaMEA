import numpy as np
from scipy.optimize import differential_evolution

class ADE_H:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.adaptation_rate = 0.05

    def __call__(self, func):
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget, self.dim))
        elite_set = population[:int(self.budget * self.elitism_ratio)]

        for _ in range(self.budget - len(elite_set)):
            fitness = np.array([func(x) for x in population])
            new_population = differential_evolution(func, self.search_space, x0=elite_set, popsize=len(elite_set) + 1, maxiter=1)

            # Adapt the elite set based on the probability
            adapt_indices = np.random.choice(len(elite_set), size=int(self.budget * self.adaptation_rate), replace=False)
            new_elite_set = elite_set[adapt_indices]
            new_elite_set = np.array([new_individual for new_individual in new_elite_set if np.random.rand() > 0.05])

            population = np.concatenate((elite_set, new_population[0:1]))
            elite_set = np.concatenate((new_elite_set, new_population[1:]))

        # Return the best solution
        return np.min(func(population))

# Example usage:
def func(x):
    return np.sum(x**2)

ade_h = ADE_H(budget=100, dim=10)
best_solution = ade_h(func)
print(f"Best solution: {best_solution}")
import numpy as np
from scipy.optimize import differential_evolution

class HEAAdHAPM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.mutation_prob = 0.05

    def __call__(self, func):
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget, self.dim))
        elite_set = population[:int(self.budget * self.elitism_ratio)]

        for _ in range(self.budget - len(elite_set)):
            fitness = np.array([func(x) for x in population])
            new_population = differential_evolution(func, self.search_space, x0=elite_set, popsize=len(elite_set) + 1, maxiter=1)

            # Adaptive probabilistic mutation
            mutation_mask = np.random.rand(len(new_population[0])) < self.mutation_prob
            new_population = np.array([new_individual if not mutation_mask else new_individual + np.random.uniform(-1, 1) for new_individual in new_population[0:1]])

            population = np.concatenate((elite_set, new_population[0:1]))
            elite_set = population[:int(self.budget * self.elitism_ratio)]

        return np.min(func(population))

# Example usage:
def func(x):
    return np.sum(x**2)

hea_apm = HEAAdHAPM(budget=100, dim=10)
best_solution = hea_apm(func)
print(f"Best solution: {best_solution}")
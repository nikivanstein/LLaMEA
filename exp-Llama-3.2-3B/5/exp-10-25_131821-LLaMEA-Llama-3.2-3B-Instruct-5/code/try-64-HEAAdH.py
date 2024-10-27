import numpy as np
from scipy.optimize import differential_evolution

class HEAAdH:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.refine_probability = 0.05

    def __call__(self, func):
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget, self.dim))

        elite_set = population[:int(self.budget * self.elitism_ratio)]

        for _ in range(self.budget - len(elite_set)):
            fitness = np.array([func(x) for x in population])

            new_population = differential_evolution(func, self.search_space, x0=elite_set, popsize=len(elite_set) + 1, maxiter=1)

            updated_population = []
            for i, individual in enumerate(population):
                if np.random.rand() < self.refine_probability:
                    # Refine individual with a probability of 0.05
                    refined_individual = individual + np.random.uniform(-0.1, 0.1, size=self.dim)
                    refined_fitness = func(refined_individual)
                    updated_population.append((refined_individual, refined_fitness))
                else:
                    updated_population.append((individual, fitness[i]))

            updated_population = sorted(updated_population, key=lambda x: x[1])

            population = np.array(updated_population)[:self.budget]

            elite_set = population[:int(self.budget * self.elitism_ratio)]

        return np.min(func(population))

# Example usage:
def func(x):
    return np.sum(x**2)

hea_adh = HEAAdH(budget=100, dim=10)
best_solution = hea_adh(func)
print(f"Best solution: {best_solution}")
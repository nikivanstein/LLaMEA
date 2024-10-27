import numpy as np
from scipy.optimize import differential_evolution

class HEAAdH:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.selection_prob = 0.05

    def __call__(self, func):
        # Initialize population with random points
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget, self.dim))

        # Initialize elite set
        elite_set = population[:int(self.budget * self.elitism_ratio)]

        # Perform probabilistic selection
        selected_population = population[np.random.choice(self.budget, size=int(self.budget * self.selection_prob), replace=False)]

        # Perform differential evolution
        for _ in range(self.budget - len(elite_set) - len(selected_population)):
            # Evaluate selected population
            fitness = np.array([func(x) for x in selected_population])

            # Perform differential evolution
            new_population = differential_evolution(func, self.search_space, x0=elite_set, popsize=len(elite_set) + len(selected_population), maxiter=1)

            # Update population and elite set
            population = np.concatenate((elite_set, selected_population, new_population[0:1]))
            elite_set = population[:int(self.budget * self.elitism_ratio)]

        # Perform probabilistic selection again
        selected_population = population[np.random.choice(self.budget, size=int(self.budget * self.selection_prob), replace=False)]

        # Return the best solution
        return np.min(func(selected_population))

# Example usage:
def func(x):
    return np.sum(x**2)

hea_adh = HEAAdH(budget=100, dim=10)
best_solution = hea_adh(func)
print(f"Best solution: {best_solution}")
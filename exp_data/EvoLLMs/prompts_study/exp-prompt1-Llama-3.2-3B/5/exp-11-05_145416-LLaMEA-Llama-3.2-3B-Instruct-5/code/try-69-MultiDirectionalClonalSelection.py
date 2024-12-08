import numpy as np
from scipy.optimize import differential_evolution

class MultiDirectionalClonalSelection:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 50

    def __call__(self, func):
        def neg_func(x):
            return -func(x)

        bounds = [(self.search_space[0], self.search_space[1]) for _ in range(self.dim)]

        # Differential Evolution optimization
        de_result = differential_evolution(neg_func, bounds, x0=np.random.uniform(self.search_space[0], self.search_space[1], size=(self.population_size, self.dim)))

        # Clonal selection
        f_values = [func(x) for x in de_result.x]
        selected_indices = np.argsort(f_values)[:int(self.budget * 0.1)]
        selected_individuals = de_result.x[selected_indices]

        # Multi-directional selection
        multi_directional_indices = np.random.choice(selected_individuals.shape[0], size=self.population_size, replace=False)
        selected_individuals = selected_individuals[multi_directional_indices]

        # Update with DE-based perturbation
        de_result_pert = differential_evolution(neg_func, bounds, x0=np.random.uniform(self.search_space[0], self.search_space[1], size=(self.population_size, self.dim)))
        de_perturbed_individuals = de_result_pert.x

        # Combine selected and perturbed individuals
        combined_individuals = np.concatenate((selected_individuals, de_perturbed_individuals))

        return combined_individuals

# Example usage
def func(x):
    return np.sum(x**2)

bbo = MultiDirectionalClonalSelection(budget=10, dim=5)
optimized_func = bbo(func)
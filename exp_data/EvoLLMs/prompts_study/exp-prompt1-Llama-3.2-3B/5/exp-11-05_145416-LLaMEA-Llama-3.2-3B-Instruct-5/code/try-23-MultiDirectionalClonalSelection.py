import numpy as np
from scipy.optimize import differential_evolution

class MultiDirectionalClonalSelection:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 50
        self.selection_rate = 0.1

    def __call__(self, func):
        def neg_func(x):
            return -func(x)

        bounds = [(self.search_space[0], self.search_space[1]) for _ in range(self.dim)]

        # Differential Evolution optimization
        de_result = differential_evolution(neg_func, bounds, x0=np.random.uniform(self.search_space[0], self.search_space[1], size=(self.population_size, self.dim)))

        # Clonal selection
        f_values = [func(x) for x in de_result.x]
        sorted_indices = np.argsort(f_values)
        selected_indices = sorted_indices[:int(self.budget * self.selection_rate)]
        selected_individuals = de_result.x[sorted_indices[:int(self.budget * self.selection_rate)]]

        # Multi-directional selection
        multi_directional_indices = np.random.choice(selected_individuals.shape[0], size=self.population_size - len(selected_indices), replace=False)
        selected_individuals = np.concatenate((selected_individuals, selected_individuals[multi_directional_indices]))

        return selected_individuals

# Example usage
def func(x):
    return np.sum(x**2)

bbo = MultiDirectionalClonalSelection(budget=10, dim=5)
optimized_func = bbo(func)
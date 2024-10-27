import numpy as np

class ChaoticMapDynamicSearchSpaceExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lower_bound = -5.0
        upper_bound = 5.0
        best_solution = np.random.uniform(lower_bound, upper_bound, size=self.dim)
        best_fitness = func(best_solution)
        step_size = 0.1 * (upper_bound - lower_bound)  # Adaptive step size
        for _ in range(self.budget):
            # Introduce chaotic maps for exploring new solutions
            chaotic_step = np.sin(best_solution) * np.cos(best_solution)  # Example of a chaotic map
            new_solution = best_solution + chaotic_step * step_size
            new_solution = np.clip(new_solution, lower_bound, upper_bound)
            new_fitness = func(new_solution)
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                step_size *= 0.95  # Self-adaptive strategy enhancement
        return best_solution
import numpy as np

class NovelDynamicSearchSpaceExploration:
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
            # Levy flights with a modified step update rule
            levy_step = np.random.standard_cauchy(size=self.dim) / np.sqrt(np.abs(np.random.normal(size=self.dim)))  
            new_solution = best_solution + levy_step * step_size
            new_solution = np.clip(new_solution, lower_bound, upper_bound)
            new_fitness = func(new_solution)
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                step_size *= 0.95  # Self-adaptive step size update
        return best_solution
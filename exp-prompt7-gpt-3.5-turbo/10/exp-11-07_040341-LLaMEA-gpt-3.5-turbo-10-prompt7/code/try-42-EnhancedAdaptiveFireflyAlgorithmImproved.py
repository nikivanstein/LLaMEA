import numpy as np

class EnhancedAdaptiveFireflyAlgorithmImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.attractiveness_values = np.exp(-np.linspace(0, 2, num=self.dim))  # Precompute attractiveness values

    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget):
            current_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            current_fitness = func(current_solution)

            if current_fitness < best_fitness:
                best_solution, best_fitness = current_solution, current_fitness

            if func(current_solution) < func(best_solution):
                best_solution = current_solution

            step_size = np.random.normal(0, 0.1, self.dim)  # Improved Exploration
            current_solution = current_solution * 0.9 + best_solution * 0.1 + step_size
            np.clip(current_solution, self.lower_bound, self.upper_bound, out=current_solution)  # Exploitation

            new_solution = current_solution * (1 - self.attractiveness_values) + best_solution * self.attractiveness_values
            if func(new_solution) < func(current_solution):
                current_solution = new_solution

        return best_solution
import numpy as np

class EnhancedAdaptiveFireflyAlgorithmImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.alpha = 0.1
        self.beta = 0.9
        self.lower_bound, self.upper_bound = self.bounds
        self.attractiveness_values = np.exp(-np.arange(self.dim) * 2 / (self.dim - 1))  # Precompute attractiveness values

    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget):
            current_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            current_fitness = func(current_solution)

            if current_fitness < best_fitness:
                best_solution, best_fitness = current_solution, current_fitness

            step_size = np.random.normal(0, 0.1, self.dim)  # Improved Exploration
            current_solution = self.beta * current_solution + self.alpha * best_solution + step_size
            np.clip(current_solution, self.lower_bound, self.upper_bound, out=current_solution)

            new_solution = current_solution * (1 - self.attractiveness_values) + best_solution * self.attractiveness_values
            if func(new_solution) < func(current_solution):
                current_solution = new_solution

        return best_solution
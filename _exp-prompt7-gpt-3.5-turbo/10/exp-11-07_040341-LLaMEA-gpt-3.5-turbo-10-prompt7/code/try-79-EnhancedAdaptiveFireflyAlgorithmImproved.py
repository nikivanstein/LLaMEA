import numpy as np

class EnhancedAdaptiveFireflyAlgorithmImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.alpha = 0.1
        self.beta = 0.9
        self.lower_bound, self.upper_bound = self.bounds
        self.attractiveness_values = np.exp(-np.linspace(0, 2, num=self.dim))  # Precompute attractiveness values

    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget):
            current_solution = np.clip(np.random.normal(0, 0.1, self.dim) + self.beta * np.random.uniform(self.lower_bound, self.upper_bound, self.dim) + self.alpha * best_solution, self.lower_bound, self.upper_bound)
            new_solution = current_solution * (1 - self.attractiveness_values) + best_solution * self.attractiveness_values
            if func(new_solution) < func(current_solution):
                current_solution = new_solution
                best_solution, best_fitness = current_solution, func(current_solution)

        return best_solution
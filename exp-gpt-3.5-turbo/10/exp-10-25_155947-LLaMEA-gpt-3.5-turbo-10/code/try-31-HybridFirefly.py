import numpy as np

class HybridFirefly:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.alpha = 0.1  # Damping coefficient
        self.beta = 0.5   # Jumping rate
        self.gamma = 0.85  # Dynamic update rate
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))

        def levy_flight():
            sigma = (np.math.gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) / np.math.gamma((1 + self.beta) / 2) * 2 ** ((self.beta - 1) / 2)) ** (1 / self.beta)
            levy = np.random.normal(0, sigma, self.dim) / np.abs(np.random.normal()) ** (1 / self.beta)
            return levy

        population = initialize_population()
        best_solution = population[np.argmin([func(individual) for individual in population])]
        
        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                new_solution = population[i] + levy_flight()
                new_solution = np.clip(new_solution, self.lb, self.ub)

                if func(new_solution) < func(population[i]):
                    population[i] = new_solution

                    if func(new_solution) < func(best_solution):
                        best_solution = new_solution

            # Dynamic update
            for i in range(self.population_size):
                if np.random.rand() < self.gamma:
                    population[i] = best_solution + self.alpha * np.random.normal(0, 1, self.dim)

        return best_solution
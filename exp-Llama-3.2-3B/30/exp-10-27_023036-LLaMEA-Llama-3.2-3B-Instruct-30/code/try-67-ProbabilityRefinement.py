import numpy as np

class ProbabilityRefinement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 100
        self.population = self.initialize_population()
        self.best_solution = None
        self.f_best = float('inf')

    def initialize_population(self):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        return population

    def evaluate(self, func):
        self.f_best = min(self.f_best, func(self.population[:, 0]))
        for i in range(self.population_size):
            self.f_best = min(self.f_best, func(self.population[i, :]))

    def __call__(self, func):
        for _ in range(self.budget):
            self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
            self.evaluate(func)

            # Select best solution
            self.best_solution = np.argmin(self.f_best)
            self.f_best = func(self.population[self.best_solution, :])

            # Refine strategy
            if self.f_best < self.f_best * 0.7:
                # 30% chance to change individual line of the selected solution
                if np.random.rand() < 0.3:
                    self.population[self.best_solution, :] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            else:
                # 30% chance to change individual line of the selected solution
                if np.random.rand() < 0.3:
                    self.population[self.best_solution, :] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    self.f_best = func(self.population[self.best_solution, :])

            # Update best solution
            self.f_best = min(self.f_best, func(self.population[:, 0]))

        return self.population[self.best_solution, :], self.f_best

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
rf = ProbabilityRefinement(budget, dim)
solution, f_best = rf(func)
print(f"Best solution: {solution}, Best fitness: {f_best}")
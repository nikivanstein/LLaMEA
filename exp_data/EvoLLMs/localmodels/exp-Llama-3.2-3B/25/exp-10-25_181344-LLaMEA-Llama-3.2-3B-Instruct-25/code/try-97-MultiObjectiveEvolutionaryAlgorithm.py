import numpy as np

class MultiObjectiveEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidate = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf
        self.objectives = []

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.candidates[:, 0])
            self.best_candidate = self.candidates[np.argmin(fitness), :]
            self.best_fitness = np.min(fitness)

            # Probabilistic Refinement
            for _ in range(10):
                new_candidate = self.candidates[np.random.choice(self.population_size, size=1, replace=False), :]
                new_fitness = func(new_candidate)
                if new_fitness < self.best_fitness:
                    self.best_candidate = new_candidate
                    self.best_fitness = new_fitness
                    self.candidates[np.argmin(fitness), :] = new_candidate

            # Evolutionary Strategy
            self.candidates[np.random.choice(self.population_size, size=10, p=[0.25, 0.75], replace=False), :] = self.candidates[np.random.choice(self.population_size, size=10, p=[0.25, 0.75], replace=False), :] + \
                                                                                      self.candidates[np.random.choice(self.population_size, size=10, p=[0.25, 0.75], replace=False), :] * \
                                                                                      np.random.uniform(-0.1, 0.1, size=(10, self.dim))

            # Selection
            self.candidates = self.candidates[np.argsort(fitness)]
            self.population_size = self.population_size // 2

            # Mutation
            self.candidates[np.random.choice(self.population_size, size=self.population_size, replace=False), :] += np.random.uniform(-0.1, 0.1, size=(self.population_size, self.dim))

            # Check if the best candidate is improved
            if self.best_fitness < func(self.best_candidate):
                self.candidates[np.argmin(fitness), :] = self.best_candidate

        return self.best_candidate, self.best_fitness

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

multi_objective_EA = MultiObjectiveEvolutionaryAlgorithm(budget=100, dim=2)
best_candidate, best_fitness = multi_objective_EA(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")
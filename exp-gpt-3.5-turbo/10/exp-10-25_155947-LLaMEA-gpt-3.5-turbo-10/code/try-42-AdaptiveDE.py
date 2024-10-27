import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))

        population = initialize_population()
        best_solution = population[np.argmin([func(individual) for individual in population])

        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                candidate = np.delete(population, i, axis=0)
                idxs = np.random.choice(range(self.population_size - 1), 3, replace=False)
                mutant = population[idxs[0]] + 0.8 * (population[idxs[1]] - population[idxs[2]])
                trial = np.clip(mutant, self.lb, self.ub)
                
                if func(trial) < func(population[i]):
                    population[i] = trial

                    if func(trial) < func(best_solution):
                        best_solution = trial

        return best_solution
import numpy as np

class DifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        evals = 0
        while evals < self.budget:
            fitness = [func(ind) for ind in population]
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]

            # Differential mutation operator
            mutated_population = population + 0.5 * (population - np.roll(population, shift=1, axis=0)) + np.random.normal(0, 0.1, (self.budget, self.dim))
            crossover_prob = np.random.rand(self.budget, self.dim) < 0.9
            population = np.where(crossover_prob, mutated_population, population)
            evals += self.budget

        return best_solution
import numpy as np

class CuckooSearchAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pa = 0.25  # Probability of a cuckoo egg being discovered

    def levy_flight(self):
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / beta))
        return 0.01 * step

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        nest_fitness = [func(ind) for ind in population]

        for _ in range(self.budget):
            new_population = population.copy()
            for i in range(len(population)):
                step = self.levy_flight()
                new_position = np.clip(population[i] + step, -5.0, 5.0)
                if func(new_position) < nest_fitness[i]:
                    nest_fitness[i] = func(new_position)
                    new_population[i] = new_position

            replace_indices = np.argsort(nest_fitness)[:int(self.pa * len(population))]
            for i in replace_indices:
                new_population[i] = np.random.uniform(-5.0, 5.0, self.dim)
                nest_fitness[i] = func(new_population[i])

            population = new_population

        best_solution = population[np.argmin(nest_fitness)]
        return best_solution
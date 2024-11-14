import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.elite_fraction = 0.1  # Fraction of elite individuals to retain

    def __call__(self, func):
        np.random.seed(0)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        evals = self.population_size

        while evals < self.budget:
            elite_count = max(1, int(self.elite_fraction * self.population_size))
            elite_indices = np.argsort(fitness)[:elite_count]
            next_generation = np.copy(population[elite_indices])

            for i in range(self.population_size):
                if i < elite_count:
                    continue
                idxs = list(set(range(self.population_size)) - {i})
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                local_crossover_rate = self.crossover_rate * (1 - (evals / self.budget))
                trial = np.where(np.random.rand(self.dim) < local_crossover_rate, mutant, population[i])

                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    next_generation = np.vstack((next_generation, trial))
                    fitness[i] = trial_fitness
                else:
                    next_generation = np.vstack((next_generation, population[i]))

                if evals >= self.budget:
                    break

            population = next_generation[:self.population_size]

        best_index = np.argmin(fitness)
        return population[best_index]
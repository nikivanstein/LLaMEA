import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 12 * dim  # Slightly increased population size
        self.initial_mutation_factor = 0.9  # Adjusted mutation factor
        self.crossover_rate = 0.8  # Slightly reduced crossover rate
        self.mutation_decay = 0.99  # Introduced decay for mutation factor

    def __call__(self, func):
        np.random.seed(0)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        evals = self.population_size
        mutation_factor = self.initial_mutation_factor

        while evals < self.budget:
            for i in range(self.population_size):
                idxs = list(range(self.population_size))
                idxs.remove(i)
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                mutant = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])
                
                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial

                if evals >= self.budget:
                    break

            mutation_factor *= self.mutation_decay  # Apply decay to mutation factor

        best_index = np.argmin(fitness)
        return population[best_index]
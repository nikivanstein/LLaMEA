import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population_size_decay_rate = 0.95

    def __call__(self, func):
        np.random.seed(0)
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        evals = population_size

        while evals < self.budget:
            reduced_population = int(population_size * self.population_size_decay_rate)
            for i in range(reduced_population):
                idxs = list(range(population_size))
                idxs.remove(i)
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])
                
                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial

                if evals >= self.budget:
                    break

            best_indices = np.argsort(fitness)[:reduced_population]
            population = population[best_indices]
            fitness = fitness[best_indices]
            population_size = reduced_population

        best_index = np.argmin(fitness)
        return population[best_index]
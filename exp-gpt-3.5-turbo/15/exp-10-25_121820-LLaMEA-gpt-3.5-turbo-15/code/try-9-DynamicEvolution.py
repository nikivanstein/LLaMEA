import numpy as np

class DynamicEvolution:
    def __init__(self, budget, dim, population_size=30, scaling_factor=0.5, crossover_rate=0.9, mutation_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.scaling_factor = scaling_factor
        self.crossover_rate = crossover_rate
        self.mutation_prob = mutation_prob

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        population = initialize_population()
        fitness = np.array([func(individual) for individual in population])
        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.scaling_factor * (b - c), -5.0, 5.0)
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover_mask, mutant, population[i])
                
                if np.random.rand() < self.mutation_prob:
                    local_search = population[i] + np.random.uniform(-0.2, 0.2, self.dim)
                    trial_fitness_local = func(local_search)
                    if trial_fitness_local < fitness[i]:
                        trial = local_search
                        trial_fitness = trial_fitness_local
                else:
                    trial_fitness = func(trial)
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

        best_idx = np.argmin(fitness)
        return population[best_idx]
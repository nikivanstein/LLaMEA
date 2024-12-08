import numpy as np

class EnhancedDifferentialEvolution(DifferentialEvolution):
    def __init__(self, budget, dim, population_size=30, scaling_factor=0.5, crossover_rate=0.9, mutation_probability=0.2):
        super().__init__(budget, dim, population_size, scaling_factor, crossover_rate)
        self.mutation_probability = mutation_probability

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
                mutation_mask = np.random.rand(self.dim) < self.mutation_probability
                mutant = np.where(mutation_mask, mutant, population[i])
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover_mask, mutant, population[i])
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

        best_idx = np.argmin(fitness)
        return population[best_idx]
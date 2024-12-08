import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Common practice for DE
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_factor = 0.5  # Initial mutation factor
        self.crossover_rate = 0.7  # Initial crossover rate

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size  # Initial evaluations done during population initialization

        while evaluations < self.budget:
            # Adaptive parameters
            diversity = np.std(population, axis=0).mean()
            self.mutation_factor = 0.5 + 0.5 * (1 - diversity / (self.upper_bound - self.lower_bound))
            self.crossover_rate = 0.9 - 0.5 * (diversity / (self.upper_bound - self.lower_bound))
            
            new_population = np.copy(population)

            for i in range(self.population_size):
                # Mutation
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.array([
                    mutant[j] if np.random.rand() < self.crossover_rate or j == np.random.randint(self.dim) else population[i][j]
                    for j in range(self.dim)
                ])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

            population = new_population

        return best_solution
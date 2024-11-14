import numpy as np

class RefinedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim  # Initial population size
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize population
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = population_size

        while evaluations < self.budget:
            # Adaptive parameters
            diversity = np.std(population, axis=0).mean()
            progress_ratio = evaluations / self.budget
            mutation_factor = 0.5 + 0.3 * (1 - progress_ratio) * (1 - diversity / (self.upper_bound - self.lower_bound))
            crossover_rate = 0.9 - 0.5 * progress_ratio

            new_population = np.copy(population)

            for i in range(population_size):
                # Mutation
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.array([
                    mutant[j] if np.random.rand() < crossover_rate or j == np.random.randint(self.dim) else population[i][j]
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

            # Adaptive population size reduction
            if evaluations < self.budget and evaluations > self.budget * 0.5:
                new_population_size = max(4, int(self.initial_population_size * (1 - progress_ratio)))
                sorted_indices = np.argsort(fitness)
                new_population = new_population[sorted_indices[:new_population_size]]
                fitness = fitness[sorted_indices[:new_population_size]]
                population_size = new_population_size

            population = new_population

        return best_solution
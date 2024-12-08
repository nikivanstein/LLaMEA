import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4 * self.dim, 20)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0

    def __call__(self, func):
        best_solution = None
        best_fitness = np.inf
        stagnation_counter = 0

        # Initial fitness evaluation
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1
            if self.fitness[i] < best_fitness:
                best_fitness = self.fitness[i]
                best_solution = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Self-adaptive mutation factor and crossover rate
                F = np.clip(0.5 + np.random.randn() * 0.3, 0.1, 0.9)
                CR = np.clip(0.5 + np.random.randn() * 0.2, 0.1, 0.9)

                # Mutation
                indices = np.random.choice([x for x in range(self.population_size) if x != i], 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.copy(self.population[i])
                crossover_points = np.random.rand(self.dim) < CR
                if not np.any(crossover_points):
                    crossover_points[np.random.randint(self.dim)] = True
                trial[crossover_points] = mutant[crossover_points]

                # Selection
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial
                        stagnation_counter = 0
                else:
                    stagnation_counter += 1

                # Population dynamics adjustment
                if stagnation_counter > self.population_size:
                    additional_population = np.random.uniform(self.lower_bound, self.upper_bound, (self.dim,))
                    additional_fitness = func(additional_population)
                    self.evaluations += 1
                    if additional_fitness < best_fitness:
                        best_fitness = additional_fitness
                        best_solution = additional_population
                    stagnation_counter = 0

                if self.evaluations >= self.budget:
                    break

        return best_solution
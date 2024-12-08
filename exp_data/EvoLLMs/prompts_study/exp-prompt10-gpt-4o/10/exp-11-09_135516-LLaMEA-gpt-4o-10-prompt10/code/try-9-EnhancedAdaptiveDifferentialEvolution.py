import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.evaluations = 0

    def __call__(self, func):
        best_solution = None
        best_fitness = np.inf

        # Initial fitness evaluation
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1
            if self.fitness[i] < best_fitness:
                best_fitness = self.fitness[i]
                best_solution = self.population[i]

        def opposition_based_learning():
            return self.lower_bound + self.upper_bound - self.population

        while self.evaluations < self.budget:
            opposition_population = opposition_based_learning()
            for i in range(self.population_size):
                if np.random.rand() < 0.5:
                    self.fitness[i] = func(opposition_population[i])
                    self.evaluations += 1
                    if self.fitness[i] < best_fitness:
                        best_fitness = self.fitness[i]
                        best_solution = opposition_population[i]
                if self.evaluations >= self.budget:
                    break

                # Adaptative mutation factor and crossover rate
                F = self.mutation_factor * (1 + np.random.normal(0, 0.1))
                CR = self.crossover_rate * (1 + np.random.normal(0, 0.1))

                if F < 0.1: F = 0.1
                if F > 1.0: F = 1.0
                if CR < 0.1: CR = 0.1
                if CR > 1.0: CR = 1.0

                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.copy(self.population[i])
                crossover_points = np.random.rand(self.dim) < CR
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
                if self.evaluations >= self.budget:
                    break
        return best_solution
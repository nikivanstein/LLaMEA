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
        self.elitism_rate = 0.1  # Introduced elitism

    def __call__(self, func):
        best_solution = None
        best_fitness = np.inf
        elite_size = max(1, int(self.elitism_rate * self.population_size))

        # Initial fitness evaluation
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1
            if self.fitness[i] < best_fitness:
                best_fitness = self.fitness[i]
                best_solution = self.population[i]

        while self.evaluations < self.budget:
            sorted_indices = np.argsort(self.fitness)
            elites = self.population[sorted_indices[:elite_size]]

            for i in range(self.population_size):
                F = self.mutation_factor + 0.1 * (1 - self.fitness[i] / (best_fitness + 1e-12))

                CR = self.crossover_rate + np.random.normal(0, 0.05)

                if F < 0: F = 0.1
                if F > 1: F = 1
                if CR < 0: CR = 0.1
                if CR > 1: CR = 1

                # Mutation with elitism influence
                indices = np.random.choice(elite_size, 3, replace=False)
                a, b, c = elites[indices]
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
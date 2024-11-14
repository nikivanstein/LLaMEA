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
        self.strategy_switch_frequency = 50  # New parameter for hybrid strategy

    def __call__(self, func):
        best_solution = None
        best_fitness = np.inf
        generation = 0  # Track the number of generations

        # Initial fitness evaluation
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1
            if self.fitness[i] < best_fitness:
                best_fitness = self.fitness[i]
                best_solution = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Adaptative mutation factor and crossover rate with decay
                F = self.mutation_factor * (0.9 ** (generation // self.strategy_switch_frequency))
                CR = self.crossover_rate * (0.9 ** (generation // self.strategy_switch_frequency))

                if F < 0.1: F = 0.1
                if CR < 0.1: CR = 0.1

                # Hybrid Mutation Strategy
                if generation % (2 * self.strategy_switch_frequency) < self.strategy_switch_frequency:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = self.population[indices]
                    mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                else:
                    indices = np.random.choice(self.population_size, 5, replace=False)
                    a, b, c, d, e = self.population[indices]
                    mutant = np.clip(a + F * (b - c + d - e), self.lower_bound, self.upper_bound)

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
            generation += 1
        return best_solution
import numpy as np

class EnhancedMixedStrategyDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4 * self.dim, 20)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0
        self.dynamic_shrinkage_rate = 0.95  # Further increase in shrinkage rate
        self.elitism_rate = 0.10  # Increased elitism rate for stronger focus on best solutions

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

        while self.evaluations < self.budget:
            # Dynamically resize population
            if self.evaluations % (self.budget // (self.population_size // 4)) == 0:
                self.population_size = int(self.population_size * self.dynamic_shrinkage_rate)
                self.fitness = self.fitness[:self.population_size]
                self.population = self.population[:self.population_size]

            elitism_count = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(self.fitness)[:elitism_count]
            elite_population = self.population[elite_indices]

            for i in range(self.population_size):
                F = 0.4 + np.random.rand() * 0.6  # Adjusted F range for adaptive mutation
                CR = 0.7 + np.random.rand() * 0.3  # Slightly reduced CR lower bound

                indices = np.random.choice([x for x in range(self.population_size) if x != i], 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)

                trial = np.copy(self.population[i])
                crossover_points = np.random.rand(self.dim) < CR
                if not np.any(crossover_points):
                    crossover_points[np.random.randint(self.dim)] = True
                trial[crossover_points] = mutant[crossover_points]

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

            # Integrating dynamic elitism: replace part of the population with best solutions
            self.population[:elitism_count] = elite_population

        return best_solution
import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4 * self.dim, 20)  # Base population size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0

    def __call__(self, func):
        best_solution = None
        best_fitness = np.inf
        adapt_interval = self.budget // 10  # Adaptation interval

        # Initial fitness evaluation
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1
            if self.fitness[i] < best_fitness:
                best_fitness = self.fitness[i]
                best_solution = self.population[i]

        while self.evaluations < self.budget:
            if self.evaluations % adapt_interval == 0 and self.evaluations > 0:
                # Dynamic adjustment of population size
                success_rate = np.mean(self.fitness < best_fitness)
                if success_rate < 0.2:
                    self.population_size = min(int(self.population_size * 1.2), self.budget - self.evaluations)
                elif success_rate > 0.4:
                    self.population_size = max(int(self.population_size * 0.8), 20)

            for i in range(self.population_size):
                # Dynamic adaptation of mutation factor and crossover rate
                F = 0.5 + np.random.rand() * 0.5  # Slightly wider range for mutation factor
                CR = 0.6 + np.random.rand() * 0.4  # Slightly wider range for crossover rate

                # Mutation
                indices = np.random.choice([x for x in range(self.population_size) if x != i], 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.copy(self.population[i])
                crossover_points = np.random.rand(self.dim) < CR
                if not np.any(crossover_points):
                    crossover_points[np.random.randint(self.dim)] = True  # Ensure at least one crossover point
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
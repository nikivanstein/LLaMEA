# Description: Enhanced Adaptive Differential Evolution with Dynamic Population Control and Adaptive Parameter Tuning for Faster Convergence.
# Code: 
import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4 * self.dim, 20)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0
        self.dynamic_shrinkage_rate = 0.92  # Increased shrinkage for faster convergence
        self.adaptive_cr = 0.5  # Adaptive crossover rate starting point
        self.adaptive_f = 0.6   # Adaptive scaling factor starting point

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
            if self.evaluations % (self.budget // (self.population_size // 5)) == 0:
                self.population_size = int(self.population_size * self.dynamic_shrinkage_rate)
                self.fitness = self.fitness[:self.population_size]
                self.population = self.population[:self.population_size]

            for i in range(self.population_size):
                # Adaptive parameter tuning
                F = self.adaptive_f + np.random.rand() * 0.2
                CR = self.adaptive_cr + np.random.rand() * 0.2

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
                    # Adjust adaptive parameters based on improvements
                    self.adaptive_f = 0.9 * self.adaptive_f + 0.1 * F
                    self.adaptive_cr = 0.9 * self.adaptive_cr + 0.1 * CR

                if self.evaluations >= self.budget:
                    break
        return best_solution
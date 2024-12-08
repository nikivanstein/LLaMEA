import numpy as np

class EnhancedFastConvergenceAlgorithm(EnhancedDynamicMutDEImprovedFast):
    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.population_size):
                a, b, c = np.random.choice(self.population_size, 3, replace=False)
                mutant = self.population[a] + self.scale_factors[i, 0] * (self.population[b] - self.population[c])
                trial = np.clip(mutant, -5.0, 5.0)

                crossover_points = np.random.rand(self.dim) < self.scale_factors[i, 1]
                trial = np.where(crossover_points, trial, self.population[i])

                if func(trial) < func(self.population[i]):
                    self.population[i] = trial
                    # Dynamic adaptation of mutation and crossover parameters
                    self.scale_factors[i] += np.array([0.1, 0.1]) if np.all(self.scale_factors[i] < 0.9) else np.array([-0.1, -0.1])
                    self.adaptive_factors[i] += np.array([0.05, 0.05]) if np.all(self.adaptive_factors[i] < 1.0) else np.array([-0.05, -0.05])

        final_fitness = [func(individual) for individual in self.population]
        best_idx = np.argmin(final_fitness)
        best_individual = self.population[best_idx]

        return best_individual
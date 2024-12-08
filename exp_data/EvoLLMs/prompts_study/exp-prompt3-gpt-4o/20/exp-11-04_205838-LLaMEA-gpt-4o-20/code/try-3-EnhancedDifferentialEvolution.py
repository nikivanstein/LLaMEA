# Description: Enhanced Differential Evolution with Adaptive Neighborhood-based Learning for Improved Convergence.
# Code:
import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.scaling_factor = 0.8
        self.crossover_rate = 0.9

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        
        cov_matrix = np.eye(self.dim)

        while num_evaluations < self.budget:
            trial_population = np.empty_like(population)
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 4, replace=False)
                x0, x1, x2, x3 = population[indices[:3]], population[indices[3]], population[best_idx]
                mutant_vector = x0 + self.scaling_factor * (x1 - x2 + x3 - x0)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial_vector = np.where(cross_points, mutant_vector, population[i])
                trial_population[i] = trial_vector

            trial_fitness = np.array([func(ind) for ind in trial_population])
            num_evaluations += self.population_size

            improvement_mask = trial_fitness < fitness
            population[improvement_mask] = trial_population[improvement_mask]
            fitness[improvement_mask] = trial_fitness[improvement_mask]

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < func(best_individual):
                best_individual = population[best_idx]

            if num_evaluations < self.budget:
                sampled_indices = np.random.choice(self.population_size, 2, replace=False)
                diff = population[sampled_indices[0]] - population[sampled_indices[1]]
                cov_matrix = (1 - 1.0/self.dim) * cov_matrix + (1.0/self.dim) * np.outer(diff, diff)

                self.scaling_factor = self.adjust_scaling_factor(cov_matrix)

        return best_individual

    def adjust_scaling_factor(self, cov_matrix):
        variance_measure = np.sum(np.diag(cov_matrix))
        return min(1.0, max(0.5, 0.8 * (1.0 + variance_measure)))
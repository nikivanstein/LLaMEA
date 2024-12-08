import numpy as np

class EnhancedHybridDEWithCovarianceAndLevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.scaling_factor = 0.9
        self.crossover_rate = 0.85
        self.dynamic_population = True

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        
        cov_matrix = np.eye(self.dim)

        while num_evaluations < self.budget:
            if self.dynamic_population:
                effective_pop_size = max(2, int(self.population_size * (1 - num_evaluations/self.budget)))
                population = population[:effective_pop_size]
                fitness = fitness[:effective_pop_size]

            trial_population = np.empty_like(population)
            for i in range(len(population)):
                indices = np.random.choice(len(population), 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant_vector = x0 + self.scaling_factor * (x1 - x2)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial_vector = np.where(cross_points, mutant_vector, population[i])
                trial_population[i] = trial_vector

            trial_fitness = np.array([func(ind) for ind in trial_population])
            num_evaluations += len(trial_population)

            improvement_mask = trial_fitness < fitness
            population[improvement_mask] = trial_population[improvement_mask]
            fitness[improvement_mask] = trial_fitness[improvement_mask]

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < func(best_individual):
                best_individual = population[best_idx]

            if num_evaluations < self.budget:
                sampled_indices = np.random.choice(len(population), 2, replace=False)
                diff = population[sampled_indices[0]] - population[sampled_indices[1]]
                cov_matrix = 0.9 * cov_matrix + 0.1 * np.outer(diff, diff)

                self.scaling_factor = self.adjust_scaling_factor(cov_matrix)

            if num_evaluations < self.budget and np.random.rand() < 0.5:
                levy_step = self.levy_flight(self.dim) * (1 + np.abs(func(best_individual)) / np.min(fitness))
                candidate = best_individual + levy_step * (best_individual - np.mean(population, axis=0))
                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                num_evaluations += 1
                if candidate_fitness < func(best_individual):
                    best_individual = candidate

        return best_individual

    def adjust_scaling_factor(self, cov_matrix):
        variance_measure = np.sum(np.diag(cov_matrix))
        return min(1.0, max(0.6, 0.9 * (1.0 + variance_measure)))

    def levy_flight(self, dim, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size=(dim,))
        v = np.random.normal(0, 1, size=(dim,))
        step = u / np.abs(v)**(1 / beta)
        return step

# Usage example:
# optimizer = EnhancedHybridDEWithCovarianceAndLevy(budget=10000, dim=10)
# best_solution = optimizer(my_black_box_function)
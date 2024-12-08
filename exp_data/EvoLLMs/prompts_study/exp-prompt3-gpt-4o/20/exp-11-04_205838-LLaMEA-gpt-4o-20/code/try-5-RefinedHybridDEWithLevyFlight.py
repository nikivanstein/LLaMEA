import numpy as np

class RefinedHybridDEWithLevyFlight:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.scaling_factor = 0.8
        self.crossover_rate = 0.9

    def __call__(self, func):
        # Initialize population
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = population_size
        
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        
        # Covariance matrix for learning
        cov_matrix = np.eye(self.dim)

        while num_evaluations < self.budget:
            # Dynamically adjust population size
            if num_evaluations > self.budget * 0.5:
                population_size = max(self.dim, population_size // 2)
                population = population[:population_size]
                fitness = fitness[:population_size]

            # Generate trial population via DE mutation and crossover
            trial_population = np.empty_like(population)
            for i in range(population_size):
                indices = np.random.choice(population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant_vector = x0 + self.scaling_factor * (x1 - x2)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial_vector = np.where(cross_points, mutant_vector, population[i])
                trial_population[i] = trial_vector

            # Evaluate trial population
            trial_fitness = np.array([func(ind) for ind in trial_population])
            num_evaluations += population_size

            # Replace if trial solution is better
            improvement_mask = trial_fitness < fitness
            population[improvement_mask] = trial_population[improvement_mask]
            fitness[improvement_mask] = trial_fitness[improvement_mask]

            # Update best individual
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < func(best_individual):
                best_individual = population[best_idx]

            # Covariance Matrix Learning Update
            if num_evaluations < self.budget:
                sampled_indices = np.random.choice(population_size, 2, replace=False)
                diff = population[sampled_indices[0]] - population[sampled_indices[1]]
                cov_matrix = (1 - 1.0/self.dim) * cov_matrix + (1.0/self.dim) * np.outer(diff, diff)

                # Adaptive adjustment of scaling factor
                self.scaling_factor = self.adjust_scaling_factor(cov_matrix)

            # Integrate Levy Flight for enhanced exploration
            if num_evaluations < self.budget and np.random.rand() < 0.5:
                levy_step = self.levy_flight(self.dim)
                candidate = best_individual + levy_step * (population[best_idx] - np.mean(population, axis=0))
                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                num_evaluations += 1
                if candidate_fitness < func(best_individual):
                    best_individual = candidate

        return best_individual

    def adjust_scaling_factor(self, cov_matrix):
        variance_measure = np.sum(np.diag(cov_matrix))
        return min(1.0, max(0.6, 0.8 * (1.0 + variance_measure)))

    def levy_flight(self, dim, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size=(dim,))
        v = np.random.normal(0, 1, size=(dim,))
        step = u / np.abs(v)**(1 / beta)
        return step

# Usage example:
# optimizer = RefinedHybridDEWithLevyFlight(budget=10000, dim=10)
# best_solution = optimizer(my_black_box_function)
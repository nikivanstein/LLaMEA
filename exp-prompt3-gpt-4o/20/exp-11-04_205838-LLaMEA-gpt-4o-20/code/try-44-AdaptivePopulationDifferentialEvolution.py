import numpy as np

class AdaptivePopulationDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Increased population size for diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_scaling_factor = 0.5  # Adjusted initial scaling factor
        self.crossover_rate = 0.8  # Adjusted crossover rate for exploration
        self.scaling_increase_step = 0.06  # Adjusted increment for adaptive tuning
        self.reinit_threshold = 0.1  # Increased re-initialization threshold

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        elite_individual = best_individual
        scaling_factor = self.initial_scaling_factor

        while num_evaluations < self.budget:
            trial_population = np.empty_like(population)
            for i in range(len(population)):
                indices = np.random.choice(len(population), 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant_vector = x0 + scaling_factor * (x1 - x2)
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
            if fitness[best_idx] < func(elite_individual):
                elite_individual = population[best_idx]
                scaling_factor = min(1.0, scaling_factor + self.scaling_increase_step)
            else:
                scaling_factor = max(0.1, scaling_factor - self.scaling_increase_step)

            if num_evaluations < self.budget and np.random.rand() < 0.2:
                candidate = elite_individual + 0.15 * (population[best_idx] - np.random.uniform(self.lower_bound, self.upper_bound, self.dim))
                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                num_evaluations += 1
                if candidate_fitness < func(elite_individual):
                    elite_individual = candidate

            if np.random.rand() < self.reinit_threshold:
                random_indices = np.random.choice(len(population), int(self.population_size * 0.20), replace=False)  # Increased diversification
                for idx in random_indices:
                    population[idx] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    fitness[idx] = func(population[idx])
                    num_evaluations += 1
                    if num_evaluations >= self.budget:
                        break

        return elite_individual
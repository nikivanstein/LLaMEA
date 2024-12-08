import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 8 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_scaling_factor = 0.5
        self.crossover_rate = 0.9  # Slightly increased for exploration
        self.scaling_increase_step = 0.15  # Adjusted to enhance adaptability
        self.diversity_threshold = 0.15  # New parameter for diversity management

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
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

            # Evaluate trial population
            trial_fitness = np.array([func(ind) for ind in trial_population])
            num_evaluations += len(trial_population)

            # Replace if trial solution is better
            improvement_mask = trial_fitness < fitness
            population[improvement_mask] = trial_population[improvement_mask]
            fitness[improvement_mask] = trial_fitness[improvement_mask]

            # Update best individual
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < func(best_individual):
                best_individual = population[best_idx]
                scaling_factor = min(1.0, scaling_factor + self.scaling_increase_step)
            else:
                scaling_factor = max(0.3, scaling_factor - self.scaling_increase_step)

            # Diversity management and dynamic local search
            diversity = np.std(population, axis=0).mean()
            if diversity < self.diversity_threshold:
                for idx in range(len(population)):
                    perturb = np.random.uniform(-0.1, 0.1, self.dim)
                    new_candidate = population[idx] + perturb
                    new_candidate = np.clip(new_candidate, self.lower_bound, self.upper_bound)
                    new_fitness = func(new_candidate)
                    num_evaluations += 1
                    if new_fitness < fitness[idx]:
                        population[idx] = new_candidate
                        fitness[idx] = new_fitness
                    if num_evaluations >= self.budget:
                        break

        return best_individual
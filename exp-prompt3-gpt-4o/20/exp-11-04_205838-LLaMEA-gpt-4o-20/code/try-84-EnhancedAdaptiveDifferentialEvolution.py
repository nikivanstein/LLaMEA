import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 8 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.scaling_factor = 0.5
        self.crossover_rate = 0.85
        self.scaling_increase_step = 0.07
        self.reinit_threshold = 0.1

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        elite_individual = best_individual

        while num_evaluations < self.budget:
            trial_population = np.empty_like(population)
            for i in range(len(population)):
                tournament_indices = np.random.choice(len(population), 5, replace=False)
                x0, x1, x2 = population[tournament_indices[np.argmin(fitness[tournament_indices[:3]])]], \
                             population[tournament_indices[3]], population[tournament_indices[4]]
                mutant_vector = x0 + self.scaling_factor * (x1 - x2)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                # Adaptive crossover rate based on diversity
                diversity = np.std(population, axis=0)
                adaptive_crossover_rate = max(0.5, self.crossover_rate - 0.1 * np.mean(diversity))
                cross_points = np.random.rand(self.dim) < adaptive_crossover_rate
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
                self.scaling_factor = min(1.0, self.scaling_factor + self.scaling_increase_step)
            else:
                self.scaling_factor = max(0.2, self.scaling_factor - self.scaling_increase_step)

            if np.random.rand() < self.reinit_threshold:
                random_indices = np.random.choice(len(population), int(self.population_size * 0.1), replace=False)
                for idx in random_indices:
                    if num_evaluations >= self.budget:
                        break
                    population[idx] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    fitness[idx] = func(population[idx])
                    num_evaluations += 1
            
            # Enhanced diversity preservation
            if np.std(fitness) < 1e-5 and num_evaluations < self.budget:
                diversity_increase_idx = np.random.choice(len(population), int(self.population_size * 0.05), replace=False)
                for idx in diversity_increase_idx:
                    if num_evaluations >= self.budget:
                        break
                    random_vector = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    population[idx] = random_vector
                    fitness[idx] = func(random_vector)
                    num_evaluations += 1

        return elite_individual
import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 8 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_scaling_factor = 0.5
        self.crossover_rate = 0.9
        self.scaling_increase_step = 0.06
        self.reinit_threshold = 0.1  # Adjusted reinitialization threshold
        self.local_search_intensity = 0.1  # New intensity factor for local search

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
                tournament_indices = np.random.choice(len(population), 5, replace=False)
                x0, x1, x2 = population[tournament_indices[np.argmin(fitness[tournament_indices[:3]])]], \
                             population[tournament_indices[3]], population[tournament_indices[4]]
                mutant_vector = x0 + scaling_factor * (x1 - x2 + (elite_individual - x0) * 0.5)
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
                scaling_factor = max(0.2, scaling_factor - self.scaling_increase_step)

            if np.random.rand() < self.reinit_threshold:
                random_indices = np.random.choice(len(population), int(self.population_size * 0.2), replace=False)
                for idx in random_indices:
                    if num_evaluations >= self.budget:
                        break
                    population[idx] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    fitness[idx] = func(population[idx])
                    num_evaluations += 1
            
            if not np.any(improvement_mask) and num_evaluations < self.budget:
                local_search_candidates = np.random.choice(len(population), int(self.population_size * self.local_search_intensity), replace=False)
                for idx in local_search_candidates:
                    perturbation = np.random.normal(0, 0.05, self.dim)
                    candidate = population[idx] + perturbation
                    candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                    candidate_fitness = func(candidate)
                    num_evaluations += 1
                    if candidate_fitness < fitness[idx]:
                        population[idx] = candidate
                        fitness[idx] = candidate_fitness

        return elite_individual
import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_scaling_factor = 0.5
        self.crossover_rate = 0.85
        self.scaling_increase_step = 0.07
        self.reinit_threshold = 0.1

        # Dynamic population sizing based on budget and dimension
        self.population_size = max(4 * dim, int(budget / 25))
        self.elite_archive = []

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
                mutant_vector = x0 + scaling_factor * (x1 - x2)  # Simplified mutation
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
            current_best_individual = population[best_idx]
            
            if fitness[best_idx] < func(elite_individual):
                elite_individual = current_best_individual
                scaling_factor = min(1.0, scaling_factor + self.scaling_increase_step)
            else:
                scaling_factor = max(0.2, scaling_factor - self.scaling_increase_step)

            # Archive elite individuals
            if len(self.elite_archive) < 3 or fitness[best_idx] < np.max([func(ind) for ind in self.elite_archive]):
                self.elite_archive.append(current_best_individual)
                self.elite_archive = sorted(self.elite_archive, key=func)[:3]

            if np.random.rand() < self.reinit_threshold:
                random_indices = np.random.choice(len(population), int(self.population_size * 0.1), replace=False)
                for idx in random_indices:
                    if num_evaluations >= self.budget:
                        break
                    population[idx] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    fitness[idx] = func(population[idx])
                    num_evaluations += 1

            # Introduce best from archive if stagnation is detected
            if not np.any(improvement_mask) and num_evaluations < self.budget:
                for elite in self.elite_archive:
                    candidate = elite + np.random.normal(0, 0.01, self.dim)
                    candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                    candidate_fitness = func(candidate)
                    num_evaluations += 1
                    if candidate_fitness < np.max(fitness):
                        worst_idx = np.argmax(fitness)
                        population[worst_idx] = candidate
                        fitness[worst_idx] = candidate_fitness

        return elite_individual
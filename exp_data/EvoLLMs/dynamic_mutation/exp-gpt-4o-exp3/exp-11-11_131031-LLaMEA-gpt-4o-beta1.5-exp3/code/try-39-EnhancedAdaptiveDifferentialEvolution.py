import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 15 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = population_size
        elite_fraction = 0.2
        local_search_prob = 0.1
        restart_threshold = 0.5
        no_improvement_counter = 0
        max_no_improvement = 50

        while evaluations < self.budget:
            progress_ratio = evaluations / self.budget
            mutation_factor = 0.7 + 0.3 * np.random.rand() * (1 - progress_ratio)
            crossover_rate = 0.6 + 0.3 * np.random.rand() * progress_ratio

            new_population = np.copy(population)

            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)

                trial = np.array([
                    mutant[j] if np.random.rand() < crossover_rate or j == np.random.randint(self.dim) else population[i][j]
                    for j in range(self.dim)
                ])

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    no_improvement_counter = 0

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness
                else:
                    no_improvement_counter += 1

                if evaluations >= self.budget:
                    break

            # Local Search
            if evaluations < self.budget * 0.7:
                local_search_indices = np.random.choice(population_size, int(local_search_prob * population_size), replace=False)
                for idx in local_search_indices:
                    candidate = new_population[idx] + np.random.normal(0, 0.1, self.dim)
                    candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                    candidate_fitness = func(candidate)
                    evaluations += 1
                    if candidate_fitness < fitness[idx]:
                        new_population[idx] = candidate
                        fitness[idx] = candidate_fitness
                        no_improvement_counter = 0

                    if candidate_fitness < best_fitness:
                        best_solution = candidate
                        best_fitness = candidate_fitness

            # Dynamic population resizing
            if evaluations < self.budget and evaluations > self.budget * 0.3:
                elite_size = max(1, int(elite_fraction * population_size))
                sorted_indices = np.argsort(fitness)
                elite_population = population[sorted_indices[:elite_size]]
                new_population_size = max(4, int(self.initial_population_size * (1 - progress_ratio)))
                new_population = np.vstack((new_population[sorted_indices[:new_population_size - elite_size]], elite_population))
                fitness = fitness[sorted_indices[:new_population_size]]
                population_size = new_population_size

            # Adaptive restart mechanism
            if no_improvement_counter > max_no_improvement:
                no_improvement_counter = 0
                new_population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, self.dim))
                fitness = np.array([func(ind) for ind in new_population])
                evaluations += self.initial_population_size
                best_idx = np.argmin(fitness)
                best_solution = new_population[best_idx]
                best_fitness = fitness[best_idx]

            population = new_population

        return best_solution
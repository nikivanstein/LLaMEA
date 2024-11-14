import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20 * dim
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
        local_search_prob = 0.15
        stagnation_threshold = 50
        improvement_threshold = 0.001

        def local_search(individual):
            candidate = individual + np.random.normal(0, 0.05, self.dim)
            candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
            return candidate

        while evaluations < self.budget:
            progress_ratio = evaluations / self.budget
            mutation_factor = 0.5 + 0.3 * np.random.rand() * (1 - progress_ratio)
            crossover_rate = 0.7 + 0.2 * np.random.rand() * progress_ratio

            new_population = np.copy(population)
            best_fitness_improvement = 0
            stagnation_counter = 0

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

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness_improvement = best_fitness - trial_fitness
                        best_fitness = trial_fitness
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1

                if evaluations >= self.budget or stagnation_counter > stagnation_threshold:
                    break

            if evaluations < self.budget * 0.8:
                local_search_indices = np.random.choice(population_size, int(local_search_prob * population_size), replace=False)
                for idx in local_search_indices:
                    candidate = local_search(new_population[idx])
                    candidate_fitness = func(candidate)
                    evaluations += 1
                    if candidate_fitness < fitness[idx]:
                        new_population[idx] = candidate
                        fitness[idx] = candidate_fitness

                        if candidate_fitness < best_fitness:
                            best_solution = candidate
                            best_fitness = candidate_fitness

            if best_fitness_improvement < improvement_threshold:
                stagnation_counter += 1
                if stagnation_counter > stagnation_threshold and evaluations < self.budget:
                    restart_size = int(0.1 * population_size)
                    restart_population = np.random.uniform(self.lower_bound, self.upper_bound, (restart_size, self.dim))
                    for idx in range(restart_size):
                        restart_fitness = func(restart_population[idx])
                        evaluations += 1
                        if restart_fitness < best_fitness:
                            best_solution = restart_population[idx]
                            best_fitness = restart_fitness
                    population[:restart_size] = restart_population
                    stagnation_counter = 0

            population = new_population

        return best_solution
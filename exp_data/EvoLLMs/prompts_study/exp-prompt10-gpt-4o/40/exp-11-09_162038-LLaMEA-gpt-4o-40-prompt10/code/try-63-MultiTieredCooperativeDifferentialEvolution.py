import numpy as np

class MultiTieredCooperativeDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_population_size = 20
        self.population_growth_rate = 1.2
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F_base = 0.8
        self.CR_base = 0.9
        self.adaptation_rate = 0.05
        self.local_search_intensity = 0.2
        self.mutation_prob = 0.5

    def __call__(self, func):
        eval_count = 0
        population_size = self.base_population_size
        best_fitness = np.inf
        best_individual = None

        while eval_count < self.budget:
            # Dynamic population scaling
            population_size = int(min(self.budget - eval_count, population_size * self.population_growth_rate))
            population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
            fitness = np.array([func(ind) for ind in population])
            eval_count += population_size

            # Update the best solution
            current_best_index = np.argmin(fitness)
            current_best_fitness = fitness[current_best_index]
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_index]

            # Differential Evolution with adaptive parameters
            for i in range(population_size):
                F = np.clip(self.F_base + self.adaptation_rate * np.random.randn(), 0, 1)
                CR = np.clip(self.CR_base + self.adaptation_rate * np.random.randn(), 0, 1)

                if np.random.rand() < self.mutation_prob:
                    indices = np.random.choice(population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                else:
                    indices = np.random.choice(population_size, 2, replace=False)
                    a, b = population[indices]
                    mutant = np.clip(best_individual + F * (a - b), self.lower_bound, self.upper_bound)

                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial

                if eval_count >= self.budget:
                    break

            # Enhanced local search around the current best
            neighborhood_size = int(self.local_search_intensity * population_size)
            local_neighbors = best_individual + np.random.normal(0, 0.02, (neighborhood_size, self.dim))
            local_neighbors = np.clip(local_neighbors, self.lower_bound, self.upper_bound)
            local_fitness = np.array([func(ind) for ind in local_neighbors])
            eval_count += len(local_neighbors)

            if np.min(local_fitness) < best_fitness:
                best_local_index = np.argmin(local_fitness)
                best_fitness = local_fitness[best_local_index]
                best_individual = local_neighbors[best_local_index]

        return best_individual
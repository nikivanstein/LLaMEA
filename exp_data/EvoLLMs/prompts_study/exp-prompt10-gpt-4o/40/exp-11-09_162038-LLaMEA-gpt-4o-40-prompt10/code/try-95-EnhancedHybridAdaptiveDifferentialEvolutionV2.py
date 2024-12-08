import numpy as np

class EnhancedHybridAdaptiveDifferentialEvolutionV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(80, self.budget // 6)  # Increased to enhance exploration
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F_base = 0.6  # Adjusted for balanced exploration and exploitation
        self.CR_base = 0.85  # Tuned to encourage diverse offspring
        self.adaptation_rate = 0.1  # Increased for more responsive parameter change
        self.local_search_intensity = 0.2  # Intensified local search
        self.mutation_prob = 0.75  # Slightly increased to balance strategies

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]

        while eval_count < self.budget:
            for i in range(self.population_size):
                F = np.clip(self.F_base + self.adaptation_rate * np.random.standard_normal(), 0.3, 0.9)
                CR = np.clip(self.CR_base + self.adaptation_rate * np.random.standard_normal(), 0.6, 1.0)

                if np.random.rand() < self.mutation_prob:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                else:
                    indices = np.random.choice(self.population_size, 2, replace=False)
                    a, b = population[indices]
                    mutant = np.clip(best_individual + F * (a - b), self.lower_bound, self.upper_bound)

                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_individual = trial
                        best_fitness = trial_fitness

                if eval_count >= self.budget:
                    break

            neighborhood_size = int(self.local_search_intensity * self.population_size)
            local_neighbors = best_individual + np.random.normal(0, 0.05, (neighborhood_size, self.dim))
            local_neighbors = np.clip(local_neighbors, self.lower_bound, self.upper_bound)
            local_fitness = np.array([func(ind) for ind in local_neighbors])
            eval_count += len(local_neighbors)

            if np.min(local_fitness) < best_fitness:
                best_local_index = np.argmin(local_fitness)
                best_individual = local_neighbors[best_local_index]
                best_fitness = local_fitness[best_local_index]

            population[0] = best_individual
            fitness[0] = best_fitness

        return best_individual
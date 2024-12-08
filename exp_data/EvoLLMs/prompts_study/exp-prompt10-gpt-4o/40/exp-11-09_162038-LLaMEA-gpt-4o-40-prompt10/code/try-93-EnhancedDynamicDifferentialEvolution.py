import numpy as np

class EnhancedDynamicDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = min(80, self.budget // 7)  # Increased initial population size for enhanced exploration
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F_base = 0.6  # Adaptively tuned F for balanced exploration-exploitation
        self.CR_base = 0.85  # Moderate CR to maintain diversity
        self.adaptation_rate = 0.1  # Enhanced adaptation for rapid parameter tuning
        self.local_search_intensity = 0.2  # Higher local search intensity targeting promising regions
        self.mutation_prob = 0.8  # Increased probability for best mutation strategy
        self.shrink_factor = 0.95  # Dynamic shrinking of the population size

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.initial_population_size

        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]

        current_population_size = self.initial_population_size

        while eval_count < self.budget:
            for i in range(current_population_size):
                F = np.clip(self.F_base + self.adaptation_rate * np.random.randn(), 0, 1)
                CR = np.clip(self.CR_base + self.adaptation_rate * np.random.randn(), 0, 1)

                if np.random.rand() < self.mutation_prob:
                    indices = np.random.choice(current_population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                else:
                    indices = np.random.choice(current_population_size, 2, replace=False)
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

            neighborhood_size = int(self.local_search_intensity * current_population_size)
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

            if eval_count < self.budget:
                current_population_size = max(5, int(self.shrink_factor * current_population_size))  # Ensures minimum population size

        return best_individual
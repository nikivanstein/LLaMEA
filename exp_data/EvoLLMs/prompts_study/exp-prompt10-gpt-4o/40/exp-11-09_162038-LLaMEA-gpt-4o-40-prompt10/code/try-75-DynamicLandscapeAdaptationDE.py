import numpy as np

class DynamicLandscapeAdaptationDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(80, self.budget // 10)  # Larger population for diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F_base = 0.6  # Initial value for differential weight
        self.CR_base = 0.9  # High crossover rate for diverse trial vectors
        self.adaptation_rate = 0.1  # More aggressive adaptation
        self.local_search_intensity = 0.2  # Increased local search intensity
        self.mutation_prob = 0.6  # Probability of using best mutation strategy
        self.early_stopping_threshold = 1e-8  # Stopping condition for minimal improvement

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]

        while eval_count < self.budget:
            for i in range(self.population_size):
                F = np.clip(self.F_base + self.adaptation_rate * np.random.randn(), 0, 1)
                CR = np.clip(self.CR_base + self.adaptation_rate * np.random.randn(), 0, 1)

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
                        if abs(best_fitness - trial_fitness) < self.early_stopping_threshold:
                            return best_individual
                        best_individual = trial
                        best_fitness = trial_fitness

                if eval_count >= self.budget:
                    break

            neighborhood_size = int(self.local_search_intensity * self.population_size)
            local_neighbors = best_individual + np.random.normal(0, 0.1, (neighborhood_size, self.dim))
            local_neighbors = np.clip(local_neighbors, self.lower_bound, self.upper_bound)
            local_fitness = np.array([func(ind) for ind in local_neighbors])
            eval_count += len(local_neighbors)

            if np.min(local_fitness) < best_fitness:
                best_local_index = np.argmin(local_fitness)
                best_individual = local_neighbors[best_local_index]
                best_fitness = local_fitness[best_local_index]

            self.dynamic_parameter_update(fitness, population)

        return best_individual

    def dynamic_parameter_update(self, fitness, population):
        # Update parameters based on landscape analysis
        diversity = np.std(population, axis=0).mean()
        if diversity < 0.1:
            self.F_base *= 0.9  # Reduce F if diversity is low
        else:
            self.F_base *= 1.1
        self.F_base = np.clip(self.F_base, 0.4, 0.9)
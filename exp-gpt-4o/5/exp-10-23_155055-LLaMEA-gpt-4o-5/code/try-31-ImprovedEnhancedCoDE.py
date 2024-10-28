import numpy as np

class ImprovedEnhancedCoDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20 + dim * 5
        self.F_min = 0.4
        self.F_max = 0.9
        self.CR = 0.9
        self.p_best_rate = 0.2
        self.mutation_strategies = [
            self.mutation_rand_1,
            self.mutation_best_1,
            self.mutation_current_to_pbest
        ]
        self.local_search_probability = 0.05  # Adjusted local search probability
        self.local_search_decay = 0.99  # Decay factor for local search probability
        self.evaluations = 0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                strategy = np.random.choice(self.mutation_strategies)
                F = self.adaptive_mutation_factor(fitness[i], best_fitness)
                mutant = strategy(population, best_individual, i, fitness, F)
                trial = self.crossover(population[i], mutant)
                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_individual = trial
                        best_fitness = trial_fitness

                if np.random.rand() < self.local_search_probability:
                    local_candidate, local_fitness = self.local_search(trial, func)
                    self.evaluations += 1
                    if local_fitness < trial_fitness:
                        population[i] = local_candidate
                        fitness[i] = local_fitness

            self.local_search_probability *= self.local_search_decay

        return best_individual, best_fitness

    def adaptive_mutation_factor(self, fitness_i, best_fitness):
        return self.F_min + (self.F_max - self.F_min) * np.exp(-5 * abs(fitness_i - best_fitness))

    def mutation_rand_1(self, population, best_individual, target_idx, fitness, F):
        a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
        mutant = a + F * (b - c)
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def mutation_best_1(self, population, best_individual, target_idx, fitness, F):
        a, b = population[np.random.choice(self.population_size, 2, replace=False)]
        mutant = best_individual + F * (a - b)
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def mutation_current_to_pbest(self, population, best_individual, target_idx, fitness, F):
        sorted_indices = np.argsort(fitness)
        p_best_idx = sorted_indices[:max(1, int(self.p_best_rate * self.population_size))]
        p_best = population[np.random.choice(p_best_idx)]
        a, b = population[np.random.choice(self.population_size, 2, replace=False)]
        mutant = population[target_idx] + F * (p_best - population[target_idx]) + F * (a - b)
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def local_search(self, candidate, func):
        local_step_size = 0.05
        neighbors = candidate + local_step_size * np.random.uniform(-1.0, 1.0, self.dim)
        neighbors = np.clip(neighbors, self.lower_bound, self.upper_bound)
        local_fitness = func(neighbors)
        return neighbors, local_fitness
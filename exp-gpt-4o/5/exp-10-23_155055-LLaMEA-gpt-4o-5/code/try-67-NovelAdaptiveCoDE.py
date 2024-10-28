import numpy as np

class NovelAdaptiveCoDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20 + dim * 5
        self.F = 0.5
        self.CR = 0.9
        self.p_best_rate = 0.2
        self.mutation_strategies = [
            self.mutation_rand_1,
            self.mutation_best_1,
            self.mutation_current_to_pbest
        ]
        self.local_search_probability = 0.1
        self.adaptive_step_size = 0.1
        self.diversity_threshold = 1e-5

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        evaluations = self.population_size
        adaptation_counter = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                strategy = np.random.choice(self.mutation_strategies)
                mutant = strategy(population, best_individual, i, fitness)
                trial = self.crossover(population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_individual = trial
                        best_fitness = trial_fitness
                        adaptation_counter = 0
                    else:
                        adaptation_counter += 1

                if np.random.rand() < self.local_search_probability:
                    local_candidate, local_fitness = self.local_search(trial, func)
                    evaluations += 1
                    if local_fitness < trial_fitness:
                        population[i] = local_candidate
                        fitness[i] = local_fitness

            self.dynamic_parameter_adjustment(adaptation_counter)

            if self.calculate_diversity(population) < self.diversity_threshold:
                self.stochastic_reinforcement(population)

        return best_individual, best_fitness

    def mutation_rand_1(self, population, best_individual, target_idx, fitness):
        a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
        mutant = a + self.F * (b - c)
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def mutation_best_1(self, population, best_individual, target_idx, fitness):
        a, b = population[np.random.choice(self.population_size, 2, replace=False)]
        mutant = best_individual + self.F * (a - b)
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def mutation_current_to_pbest(self, population, best_individual, target_idx, fitness):
        sorted_indices = np.argsort(fitness)
        p_best_idx = sorted_indices[:max(1, int(self.p_best_rate * self.population_size))]
        p_best = population[np.random.choice(p_best_idx)]
        a, b = population[np.random.choice(self.population_size, 2, replace=False)]
        mutant = population[target_idx] + self.F * (p_best - population[target_idx]) + self.F * (a - b)
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def local_search(self, candidate, func):
        local_step_size = self.adaptive_step_size
        neighbors = candidate + local_step_size * np.random.uniform(-1.0, 1.0, self.dim)
        neighbors = np.clip(neighbors, self.lower_bound, self.upper_bound)
        local_fitness = func(neighbors)
        return neighbors, local_fitness

    def calculate_diversity(self, population):
        centroid = np.mean(population, axis=0)
        diversity = np.mean(np.linalg.norm(population - centroid, axis=1))
        return diversity

    def stochastic_reinforcement(self, population):
        indices_to_reinitialize = np.random.choice(self.population_size, size=int(0.2 * self.population_size), replace=False)
        for idx in indices_to_reinitialize:
            population[idx] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

    def dynamic_parameter_adjustment(self, adaptation_counter):
        if adaptation_counter < 5:
            self.adaptive_step_size = min(0.5, self.adaptive_step_size * 1.1)
            self.F = min(0.9, self.F * 1.05)
        else:
            self.adaptive_step_size = max(0.01, self.adaptive_step_size * 0.9)
            self.F = max(0.1, self.F * 0.95)
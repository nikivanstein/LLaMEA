import numpy as np

class EnhancedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.num_evaluations = 0
        self.diversity_factor = 0.7
        self.mutation_factor = 0.6
        self.crossover_rate = 0.8
        self.local_search_rate = 0.25
        self.adaptive_rate = 0.20

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.num_evaluations += 1

    def select_parents_random(self):
        return np.random.choice(self.population_size, 3, replace=False)

    def crossover(self, target, mutant):
        mask = np.random.rand(self.dim) < self.crossover_rate
        return np.where(mask, mutant, target)

    def mutation_de(self, rand1, rand2, rand3):
        feedback_factor = (self.fitness.mean() - self.fitness.min()) / (self.fitness.max() - self.fitness.min() + 1e-9)
        adjusted_mutation = self.mutation_factor * feedback_factor
        mutant_vector = rand1 + self.diversity_factor * adjusted_mutation * (rand2 - rand3)
        return np.clip(mutant_vector, self.lower_bound, self.upper_bound)

    def self_induced_diversity(self, individual):
        noise_scale = np.random.uniform(0.05, 0.2)
        noise = np.random.normal(0, noise_scale, self.dim)
        return np.clip(individual + noise, self.lower_bound, self.upper_bound)

    def adapt_parameters(self):
        self.mutation_factor = np.random.uniform(0.5, 0.9)
        self.crossover_rate = np.random.uniform(0.5, 0.9)
        fitness_std = np.std(self.fitness)
        self.diversity_factor = 0.8 if fitness_std < 1e-3 else 0.7
        self.adaptive_rate = 0.2 if fitness_std < 1e-3 else 0.25

    def optimize(self, func):
        self.evaluate_population(func)
        while self.num_evaluations < self.budget:
            for i in range(self.population_size):
                if self.num_evaluations >= self.budget:
                    break
                idxs = self.select_parents_random()
                rand1, rand2, rand3 = self.population[idxs[0]], self.population[idxs[1]], self.population[idxs[2]]
                mutant = self.mutation_de(rand1, rand2, rand3)
                offspring = self.crossover(self.population[i], mutant)
                offspring_fitness = func(offspring)
                self.num_evaluations += 1
                if offspring_fitness < self.fitness[i]:
                    self.population[i] = offspring
                    self.fitness[i] = offspring_fitness
                if np.random.rand() < self.local_search_rate:
                    local_candidate = self.self_induced_diversity(self.population[i])
                    local_fitness = func(local_candidate)
                    self.num_evaluations += 1
                    if local_fitness < self.fitness[i]:
                        self.population[i] = local_candidate
                        self.fitness[i] = local_fitness
            self.adapt_parameters()
            if np.random.rand() < 0.1:
                self.population_size = max(20, int(self.population_size * 0.9))
                self.population = self.population[:self.population_size]
                self.fitness = self.fitness[:self.population_size]

    def __call__(self, func):
        self.optimize(func)
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]
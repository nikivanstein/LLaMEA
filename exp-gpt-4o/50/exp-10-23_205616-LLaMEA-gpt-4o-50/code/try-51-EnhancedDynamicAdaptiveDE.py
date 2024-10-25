import numpy as np

class EnhancedDynamicAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.num_evaluations = 0
        self.diversity_factor = 0.65
        self.mutation_factor = 0.5
        self.crossover_rate = 0.85
        self.local_search_rate = 0.3
        self.adaptive_rate = 0.35

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.num_evaluations += 1

    def select_parents_random(self):
        return np.random.choice(self.population_size, 4, replace=False)

    def crossover(self, target, mutant):
        mask = np.random.rand(self.dim) < self.adaptive_rate
        offspring = np.where(mask, mutant, target)
        return np.clip(offspring, self.lower_bound, self.upper_bound)

    def mutation_de(self, rand1, rand2, rand3, rand4):
        mutant_vector = rand1 + self.diversity_factor * self.mutation_factor * (rand2 - rand3) + \
                        self.diversity_factor * (rand4 - rand1)
        return np.clip(mutant_vector, self.lower_bound, self.upper_bound)

    def self_induced_diversity(self, individual):
        noise_scale = np.random.uniform(0.05, 0.25)
        noise = np.random.normal(0, noise_scale, self.dim)
        return np.clip(individual + noise, self.lower_bound, self.upper_bound)

    def adapt_parameters(self):
        self.mutation_factor = np.random.uniform(0.2, 0.9)
        fitness_std = np.std(self.fitness)
        self.diversity_factor = 0.7 if fitness_std < 1e-3 else 0.65
        self.local_search_rate = 0.35 if fitness_std < 1e-3 else 0.3

    def optimize(self, func):
        self.evaluate_population(func)
        while self.num_evaluations < self.budget:
            best_idx = np.argmin(self.fitness)
            best = self.population[best_idx]
            for i in range(self.population_size):
                if self.num_evaluations >= self.budget:
                    break
                idxs = self.select_parents_random()
                rand1, rand2, rand3, rand4 = self.population[idxs[0]], self.population[idxs[1]], self.population[idxs[2]], self.population[idxs[3]]
                mutant = self.mutation_de(rand1, rand2, rand3, rand4)
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

    def __call__(self, func):
        self.optimize(func)
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]
import numpy as np

class EnhancedAdaptiveMemeticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 60  # Increased population size for greater diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.num_evaluations = 0
        self.diversity_factor = 0.75  # Refined for better exploration vs exploitation balance
        self.mutation_factor = 0.8  # Adjusted for improved global search ability
        self.crossover_rate = 0.9  # Increased for more recombination
        self.local_search_rate = 0.5  # Further increased to exploit local minima effectively
        self.adaptive_rate = 0.4  # Modified for better adaptation to current landscape

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
        mutant_vector = rand1 + self.diversity_factor * self.mutation_factor * (rand2 - rand3)
        return np.clip(mutant_vector, self.lower_bound, self.upper_bound)

    def self_induced_diversity(self, individual):
        noise_scale = np.random.uniform(0.25, 0.6)  # Broadened noise scale for more comprehensive exploration
        noise = np.random.normal(0, noise_scale, self.dim)
        return np.clip(individual + noise, self.lower_bound, self.upper_bound)

    def adapt_parameters(self):
        self.mutation_factor = np.random.uniform(0.65, 0.9)  # Wider range for mutation factor to enhance adaptability
        self.crossover_rate = np.random.uniform(0.8, 0.95)  # Close monitoring of crossover rates for optimization
        fitness_std = np.std(self.fitness)
        self.diversity_factor = 0.9 if fitness_std < 1e-4 else 0.7  # Updated condition for recalibration
        self.adaptive_rate = 0.3 if fitness_std < 1e-4 else 0.5  # Adjusted rates for improved learning

    def dynamic_neighborhood_local_search(self, individual, func):
        neighborhood_size = 3
        best_candidate = individual
        best_fitness = func(individual)
        self.num_evaluations += 1
        for _ in range(neighborhood_size):
            candidate = individual + np.random.uniform(-0.1, 0.1, self.dim)
            candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
            candidate_fitness = func(candidate)
            self.num_evaluations += 1
            if candidate_fitness < best_fitness:
                best_fitness = candidate_fitness
                best_candidate = candidate
        return best_candidate, best_fitness

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
                    local_candidate, local_fitness = self.dynamic_neighborhood_local_search(self.population[i], func)
                    if local_fitness < self.fitness[i]:
                        self.population[i] = local_candidate
                        self.fitness[i] = local_fitness
            self.adapt_parameters()
            if np.random.rand() < 0.25:  # Reduced chance for dynamic reduction based on performance
                self.population_size = max(20, int(self.population_size * 0.85))  # Adjusted population size reduction
                self.population = self.population[:self.population_size]
                self.fitness = self.fitness[:self.population_size]

    def __call__(self, func):
        self.optimize(func)
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]
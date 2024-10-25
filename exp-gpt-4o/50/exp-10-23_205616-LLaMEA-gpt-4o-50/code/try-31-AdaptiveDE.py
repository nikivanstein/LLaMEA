import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.num_evaluations = 0
        self.diversity_threshold = 1e-3  # Changed value for diversity threshold
        self.mutation_factor = 0.8      # Adjusted mutation factor
        self.crossover_rate = 0.8       # Adjusted crossover rate
        self.local_search_rate = 0.2    # Adjusted local search rate
        self.divergence_control = 0.3   # Adjusted divergence control parameter

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.num_evaluations += 1

    def select_parents_random(self):
        return np.random.choice(self.population_size, 3, replace=False)

    def crossover(self, target, mutant):
        mask = np.random.rand(self.dim) < self.crossover_rate
        offspring = np.where(mask, mutant, target)
        return offspring

    def mutation_de(self, rand1, rand2, rand3):
        mutant_vector = rand1 + self.mutation_factor * (rand2 - rand3) + self.divergence_control * np.random.randn(self.dim)
        return np.clip(mutant_vector, self.lower_bound, self.upper_bound)

    def stochastic_adaptive_mutation(self, individual):
        noise_scale = np.random.uniform(0.05, 0.15)  # Adjusted noise scale
        noise = np.random.normal(0, noise_scale, self.dim)
        return np.clip(individual + noise, self.lower_bound, self.upper_bound)

    def adapt_parameters(self):
        self.mutation_factor = np.random.uniform(0.5, 0.9)
        self.crossover_rate = np.random.uniform(0.7, 0.95)
        fitness_std = np.std(self.fitness)
        self.divergence_control = 0.3 if fitness_std < self.diversity_threshold else 0.1

    def introduce_random_walk(self, individual):
        step_size = np.random.uniform(0.01, 0.05)
        walk = np.random.uniform(-step_size, step_size, self.dim)
        return np.clip(individual + walk, self.lower_bound, self.upper_bound)

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
                    local_candidate = self.stochastic_adaptive_mutation(self.population[i])
                    local_fitness = func(local_candidate)
                    self.num_evaluations += 1
                    if local_fitness < self.fitness[i]:
                        self.population[i] = local_candidate
                        self.fitness[i] = local_fitness
                if np.random.rand() < 0.5:  # Introduce random walk with a probability
                    random_walk_candidate = self.introduce_random_walk(self.population[i])
                    random_walk_fitness = func(random_walk_candidate)
                    self.num_evaluations += 1
                    if random_walk_fitness < self.fitness[i]:
                        self.population[i] = random_walk_candidate
                        self.fitness[i] = random_walk_fitness
            self.adapt_parameters()

    def __call__(self, func):
        self.optimize(func)
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]
import numpy as np

class EnhancedDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, self.dim))
        self.fitness = np.full(self.initial_population_size, np.inf)
        self.num_evaluations = 0
        self.diversity_factor = 0.6
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.local_search_rate = 0.3
        self.divergence_control = 0.2
        self.adaptive_learning_rate = 0.1
    
    def evaluate_population(self, func):
        for i in range(len(self.population)):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.num_evaluations += 1

    def select_parents_random(self):
        return np.random.choice(len(self.population), 3, replace=False)

    def crossover(self, target, mutant):
        mask = np.random.rand(self.dim) < self.crossover_rate
        offspring = np.where(mask, mutant, target)
        return offspring

    def mutation_de(self, rand1, rand2, rand3):
        mutant_vector = rand1 + self.diversity_factor * self.mutation_factor * (rand2 - rand3)
        return np.clip(mutant_vector, self.lower_bound, self.upper_bound)

    def stochastic_adaptive_mutation(self, individual):
        noise_scale = np.random.uniform(0.05, 0.2)
        noise = np.random.normal(0, noise_scale, self.dim)
        return np.clip(individual + noise, self.lower_bound, self.upper_bound)

    def adapt_parameters(self):
        self.mutation_factor = np.random.uniform(0.4, 0.9)
        self.crossover_rate = np.random.uniform(0.8, 1.0)
        fitness_std = np.std(self.fitness)
        self.diversity_factor = 1.0 if fitness_std < 1e-5 else 0.6
        self.divergence_control = 0.1 if fitness_std < 1e-5 else 0.2
        self.adaptive_learning_rate = 0.1 + 0.2 * (1 - fitness_std / (np.max(self.fitness) + 1e-9))

    def reduce_population(self):
        sorted_indices = np.argsort(self.fitness)
        cutoff = int(0.8 * len(self.population))
        self.population = self.population[sorted_indices[:cutoff]]
        self.fitness = self.fitness[sorted_indices[:cutoff]]

    def optimize(self, func):
        self.evaluate_population(func)
        while self.num_evaluations < self.budget:
            if len(self.population) > 20 and np.random.rand() < 0.1:
                self.reduce_population()
            for i in range(len(self.population)):
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
            self.adapt_parameters()

    def __call__(self, func):
        self.optimize(func)
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]
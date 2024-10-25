import numpy as np

class ImprovedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.num_evaluations = 0
        self.diversity_factor = 0.5
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.num_evaluations += 1

    def select_parents_best_and_random(self):
        best_idx = np.argmin(self.fitness)
        candidate_indices = np.random.choice(self.population_size, 2, replace=False)
        return [best_idx] + candidate_indices.tolist()

    def crossover(self, target, mutant):
        mask = np.random.rand(self.dim) < self.crossover_rate
        offspring = np.where(mask, mutant, target)
        offspring = np.clip(offspring, self.lower_bound, self.upper_bound)
        return offspring

    def mutation_de(self, rand1, rand2, rand3):
        mutant_vector = rand1 + self.diversity_factor * self.mutation_factor * (rand2 - rand3)
        return np.clip(mutant_vector, self.lower_bound, self.upper_bound)

    def adapt_parameters(self):
        self.mutation_factor = np.random.uniform(0.4, 0.9)
        self.crossover_rate = np.random.uniform(0.8, 1.0)
        fitness_std = np.std(self.fitness)
        self.diversity_factor = 0.7 if fitness_std < 1e-5 else 0.5
        if np.random.rand() < 0.3:
            new_size = np.random.randint(self.population_size // 2, self.population_size)
            self.population = self.population[:new_size]
            self.fitness = self.fitness[:new_size]
            self.population_size = new_size

    def optimize(self, func):
        self.evaluate_population(func)
        while self.num_evaluations < self.budget:
            for i in range(self.population_size):
                if self.num_evaluations >= self.budget:
                    break
                idxs = self.select_parents_best_and_random()
                rand1, rand2, rand3 = self.population[idxs[0]], self.population[idxs[1]], self.population[idxs[2]]
                mutant = self.mutation_de(rand1, rand2, rand3)
                offspring = self.crossover(self.population[i], mutant)
                offspring_fitness = func(offspring)
                self.num_evaluations += 1
                if offspring_fitness < self.fitness[i]:
                    self.population[i] = offspring
                    self.fitness[i] = offspring_fitness
            self.adapt_parameters()

    def __call__(self, func):
        self.optimize(func)
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]
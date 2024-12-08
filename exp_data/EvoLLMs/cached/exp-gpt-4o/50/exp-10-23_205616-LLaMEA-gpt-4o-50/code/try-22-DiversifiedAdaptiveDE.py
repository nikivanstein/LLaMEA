import numpy as np

class DiversifiedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.num_evaluations = 0
        self.diversity_factor = 0.6
        self.mutation_factor = 0.8
        self.crossover_rate = 0.85

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.num_evaluations += 1

    def select_parents_best(self):
        indices = np.argsort(self.fitness)
        return indices[:3]

    def crossover(self, target, mutant):
        mask = np.random.rand(self.dim) < self.crossover_rate
        offspring = np.where(mask, mutant, target)
        return offspring

    def mutation_de(self, best, rand2, rand3):
        mutant_vector = best + self.diversity_factor * self.mutation_factor * (rand2 - rand3)
        return np.clip(mutant_vector, self.lower_bound, self.upper_bound)

    def adapt_parameters(self):
        self.mutation_factor = np.random.uniform(0.5, 0.95)
        self.crossover_rate = np.random.uniform(0.7, 1.0)
        fitness_std = np.std(self.fitness)
        self.diversity_factor = 0.8 if fitness_std < 1e-4 else 0.6

    def optimize(self, func):
        self.evaluate_population(func)
        while self.num_evaluations < self.budget:
            best_idx = np.argmin(self.fitness)
            best = self.population[best_idx]
            for i in range(self.population_size):
                if self.num_evaluations >= self.budget:
                    break
                idxs = self.select_parents_best()
                rand2, rand3 = self.population[idxs[1]], self.population[idxs[2]]
                mutant = self.mutation_de(best, rand2, rand3)
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
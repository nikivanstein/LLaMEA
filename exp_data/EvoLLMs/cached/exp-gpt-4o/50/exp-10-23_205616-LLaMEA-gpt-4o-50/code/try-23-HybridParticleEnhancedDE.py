import numpy as np

class HybridParticleEnhancedDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocity = np.zeros((self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.num_evaluations = 0
        self.diversity_factor = 0.5
        self.mutation_factor = 0.6
        self.crossover_rate = 0.9
        self.personal_best = np.copy(self.population)
        self.personal_best_fitness = np.full(self.population_size, np.inf)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.num_evaluations += 1
                if self.fitness[i] < self.personal_best_fitness[i]:
                    self.personal_best[i] = self.population[i]
                    self.personal_best_fitness[i] = self.fitness[i]

    def select_parents_random(self):
        return np.random.choice(self.population_size, 3, replace=False)

    def crossover(self, target, mutant):
        mask = np.random.rand(self.dim) < self.crossover_rate
        offspring = np.where(mask, mutant, target)
        return offspring

    def mutation_de(self, rand1, rand2, rand3):
        mutant_vector = rand1 + self.diversity_factor * self.mutation_factor * (rand2 - rand3)
        return np.clip(mutant_vector, self.lower_bound, self.upper_bound)

    def adapt_parameters(self):
        self.mutation_factor = np.random.uniform(0.5, 0.9)
        self.crossover_rate = np.random.uniform(0.75, 1.0)
        fitness_std = np.std(self.fitness)
        self.diversity_factor = 0.7 if fitness_std < 1e-4 else 0.5

    def optimize(self, func):
        self.evaluate_population(func)
        global_best_idx = np.argmin(self.personal_best_fitness)
        global_best = self.personal_best[global_best_idx]
        while self.num_evaluations < self.budget:
            for i in range(self.population_size):
                if self.num_evaluations >= self.budget:
                    break
                idxs = self.select_parents_random()
                rand1, rand2, rand3 = self.population[idxs[0]], self.population[idxs[1]], self.population[idxs[2]]
                mutant = self.mutation_de(rand1, rand2, rand3)
                offspring = self.crossover(self.population[i], mutant)
                self.velocity[i] = 0.5 * self.velocity[i] + 0.5 * (self.personal_best[i] - self.population[i]) + 0.5 * (global_best - self.population[i])
                offspring += self.velocity[i]
                offspring = np.clip(offspring, self.lower_bound, self.upper_bound)
                offspring_fitness = func(offspring)
                self.num_evaluations += 1
                if offspring_fitness < self.fitness[i]:
                    self.population[i] = offspring
                    self.fitness[i] = offspring_fitness
                    if offspring_fitness < self.personal_best_fitness[i]:
                        self.personal_best[i] = offspring
                        self.personal_best_fitness[i] = offspring_fitness
            global_best_idx = np.argmin(self.personal_best_fitness)
            global_best = self.personal_best[global_best_idx]
            self.adapt_parameters()

    def __call__(self, func):
        self.optimize(func)
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]
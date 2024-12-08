import numpy as np

class AdaptiveHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 60  # Increased initial population size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.num_evaluations = 0
        self.diversity_factor = 0.6  # Adjusted diversity factor for better exploration
        self.mutation_factor = 0.7  # Adjusted mutation factor
        self.crossover_rate = 0.8  # Adjusted crossover rate

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.num_evaluations += 1

    def select_parents_tournament(self):
        indices = np.random.choice(self.population_size, 4, replace=False)
        best_1 = indices[np.argmin(self.fitness[indices[:2]])]
        best_2 = indices[np.argmin(self.fitness[indices[2:]])]
        return best_1, best_2, np.random.choice(indices)

    def crossover(self, target, mutant):
        mask = np.random.rand(self.dim) < self.crossover_rate
        offspring = np.where(mask, mutant, target)
        return offspring

    def mutation_de(self, best, rand1, rand2):
        mutant_vector = best + self.diversity_factor * self.mutation_factor * (rand1 - rand2)
        return np.clip(mutant_vector, self.lower_bound, self.upper_bound)

    def adapt_parameters(self):
        fitness_std = np.std(self.fitness)
        self.mutation_factor = np.random.uniform(0.5, 0.9) if fitness_std < 1e-5 else np.random.uniform(0.4, 0.8)
        self.crossover_rate = np.random.uniform(0.7, 1.0)
        self.diversity_factor = 1.0 if fitness_std < 1e-5 else 0.6

    def optimize(self, func):
        self.evaluate_population(func)
        while self.num_evaluations < self.budget:
            best_idx = np.argmin(self.fitness)
            best = self.population[best_idx]
            for i in range(self.population_size):
                if self.num_evaluations >= self.budget:
                    break
                idx1, idx2, idx3 = self.select_parents_tournament()
                best_parent = self.population[idx1]
                rand1, rand2 = self.population[idx2], self.population[idx3]
                mutant = self.mutation_de(best_parent, rand1, rand2)
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
import numpy as np

class HybridGADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crossover_rate = 0.7
        self.mutation_factor = 0.8
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.num_evaluations = 0

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.num_evaluations += 1

    def select_parents_tournament(self, num_parents):
        selected_parents = []
        for _ in range(num_parents):
            contenders = np.random.choice(self.population_size, size=3, replace=False)
            best = min(contenders, key=lambda idx: self.fitness[idx])
            selected_parents.append(self.population[best])
        return np.array(selected_parents)

    def crossover(self, parent1, parent2):
        mask = np.random.rand(self.dim) < self.crossover_rate
        offspring = np.where(mask, parent1, parent2)
        return offspring

    def mutation_de(self, target, best, rand1, rand2):
        mutant_vector = best + self.mutation_factor * (rand1 - rand2)
        return np.clip(mutant_vector, self.lower_bound, self.upper_bound)

    def optimize(self, func):
        self.evaluate_population(func)
        while self.num_evaluations < self.budget:
            parents = self.select_parents_tournament(self.population_size)
            best_idx = np.argmin(self.fitness)
            best = self.population[best_idx]
            for i in range(self.population_size):
                if self.num_evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                rand1, rand2 = np.random.choice(idxs, 2, replace=False)
                mutant = self.mutation_de(parents[i], best, self.population[rand1], self.population[rand2])
                offspring = self.crossover(parents[i], mutant)
                offspring_fitness = func(offspring)
                self.num_evaluations += 1
                if offspring_fitness < self.fitness[i]:
                    self.population[i] = offspring
                    self.fitness[i] = offspring_fitness

    def __call__(self, func):
        self.optimize(func)
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]
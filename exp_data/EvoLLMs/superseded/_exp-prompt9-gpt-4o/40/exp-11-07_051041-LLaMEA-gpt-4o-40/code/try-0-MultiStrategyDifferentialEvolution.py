import numpy as np

class MultiStrategyDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.cross_prob = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_value = float('inf')
        self.evaluations = 0

    def __call__(self, func):
        fitness = np.apply_along_axis(func, 1, self.population)
        self.best_value = np.min(fitness)
        self.best_solution = self.population[np.argmin(fitness)]
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            next_generation = []

            for i in range(self.population_size):
                if np.random.rand() < 0.5:
                    next_generation.append(self.de_variant_1(i, fitness, func))
                else:
                    next_generation.append(self.de_variant_2(i, fitness, func))

            self.population = np.array(next_generation)
            fitness = np.apply_along_axis(func, 1, self.population)
            self.evaluations += self.population_size

            current_best_value = np.min(fitness)
            if current_best_value < self.best_value:
                self.best_value = current_best_value
                self.best_solution = self.population[np.argmin(fitness)]

        return self.best_solution

    def de_variant_1(self, index, fitness, func):
        idxs = [idx for idx in range(self.population_size) if idx != index]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
        return self.crossover(self.population[index], mutant)

    def de_variant_2(self, index, fitness, func):
        idxs = [idx for idx in range(self.population_size) if idx != index]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant = np.clip(a + self.mutation_factor * (c - b), self.lower_bound, self.upper_bound)
        return self.crossover(self.population[index], mutant)
    
    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.cross_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial
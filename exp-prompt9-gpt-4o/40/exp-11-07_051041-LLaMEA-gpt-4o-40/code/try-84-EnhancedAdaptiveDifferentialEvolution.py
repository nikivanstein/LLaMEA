import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = max(7 * dim, 35)  # Further increased dynamic population size for diversity
        self.mutation_factor = 0.6  # Adjusted mutation factor for better balance
        self.cross_prob = 0.9  # Higher crossover probability
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, dim))
        self.best_solution = None
        self.best_value = float('inf')
        self.evaluations = 0
        self.diversity_threshold = 0.15  # New diversity threshold

    def __call__(self, func):
        fitness = np.apply_along_axis(func, 1, self.population)
        best_idx = np.argmin(fitness)
        self.best_value = fitness[best_idx]
        self.best_solution = self.population[best_idx]
        self.evaluations += self.pop_size

        while self.evaluations < self.budget:
            next_gen = np.empty((self.pop_size, self.dim))
            for i in range(self.pop_size):
                if np.random.rand() < 0.5:  # Balanced probability for variant selection
                    next_gen[i] = self.variant_best(i, fitness)
                else:
                    next_gen[i] = self.variant_rand1(i)

            self.population = next_gen
            fitness = np.apply_along_axis(func, 1, self.population)
            current_best_idx = np.argmin(fitness)
            current_best_value = fitness[current_best_idx]

            if current_best_value < self.best_value:
                self.best_value = current_best_value
                self.best_solution = self.population[current_best_idx]

            self.apply_diversity_mechanism()
            self.evaluations += self.pop_size

        return self.best_solution

    def variant_best(self, index, fitness):
        best_idx = np.argmin(fitness)
        idxs = np.delete(np.arange(self.pop_size), index)
        a, b = self.population[np.random.choice(idxs, 2, replace=False)]
        mutant = np.clip(self.population[best_idx] + self.mutation_factor * (a - b), self.lb, self.ub)
        return self.crossover(self.population[index], mutant)

    def variant_rand1(self, index):
        idxs = np.delete(np.arange(self.pop_size), index)
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant = np.clip(a + self.mutation_factor * (b - c), self.lb, self.ub)
        return self.crossover(self.population[index], mutant)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.cross_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial
    
    def apply_diversity_mechanism(self):  # New method to maintain population diversity
        if np.random.rand() < self.diversity_threshold:
            random_idx = np.random.choice(range(self.pop_size))
            self.population[random_idx] = np.random.uniform(self.lb, self.ub, self.dim)
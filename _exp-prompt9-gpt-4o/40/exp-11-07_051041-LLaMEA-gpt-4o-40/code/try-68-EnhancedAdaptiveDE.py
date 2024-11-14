import numpy as np

class EnhancedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = max(4 * dim, 25)  # Adjusted population size
        self.mutation_factor = 0.6  # Updated mutation factor
        self.cross_prob = 0.9  # Increased crossover probability
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, dim))
        self.best_solution = None
        self.best_value = float('inf')
        self.evaluations = 0
        self.success_rate_threshold = 0.2  # New threshold for success rate

    def __call__(self, func):
        fitness = np.apply_along_axis(func, 1, self.population)
        best_idx = np.argmin(fitness)
        self.best_value = fitness[best_idx]
        self.best_solution = self.population[best_idx]
        self.evaluations += self.pop_size
        success_count = 0

        while self.evaluations < self.budget:
            next_gen = np.empty((self.pop_size, self.dim))
            for i in range(self.pop_size):
                strategy_prob = np.random.rand()
                if strategy_prob < 0.5:
                    next_gen[i] = self.variant_best(i, fitness)
                else:
                    next_gen[i] = self.variant_rand2(i)  # Refined strategy variant

            self.population = next_gen
            fitness = np.apply_along_axis(func, 1, self.population)
            current_best_idx = np.argmin(fitness)
            current_best_value = fitness[current_best_idx]

            if current_best_value < self.best_value:
                self.best_value = current_best_value
                self.best_solution = self.population[current_best_idx]
                success_count += 1

            self.population[np.random.choice(range(self.pop_size))] = self.best_solution
            self.evaluations += self.pop_size

            # Monitor success rate and adapt
            if success_count / self.pop_size < self.success_rate_threshold:
                self.mutation_factor *= 1.1
            success_count = 0

        return self.best_solution

    def variant_best(self, index, fitness):
        best_idx = np.argmin(fitness)
        idxs = np.delete(np.arange(self.pop_size), index)
        a, b = self.population[np.random.choice(idxs, 2, replace=False)]
        mutant = np.clip(self.population[best_idx] + self.mutation_factor * (a - b), self.lb, self.ub)
        return self.crossover(self.population[index], mutant)

    def variant_rand2(self, index):
        idxs = np.delete(np.arange(self.pop_size), index)
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant = np.clip(c + self.mutation_factor * (a - b), self.lb, self.ub)
        return self.crossover(self.population[index], mutant)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.cross_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial
import numpy as np

class EnhancedFlockingOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w = 0.5
        self.c1 = 0.8
        self.c2 = 0.9

    def _initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

    def _fitness(self, population, func):
        return np.apply_along_axis(func, 1, population)

    def _update_position(self, population, best_position):
        r1, r2 = np.random.rand(self.budget, self.dim), np.random.rand(self.budget, self.dim)
        velocity = self.w * population + self.c1 * r1 * (best_position - population) + self.c2 * r2 * (population - best_position)
        return np.clip(population + velocity, -5.0, 5.0)

    def __call__(self, func):
        population = self._initialize_population()
        best_position = population[np.argmin(self._fitness(population, func))]
        
        for _ in range(self.budget):
            self.w = max(0.4, 0.5 - _ / self.budget)  # Dynamic inertia weight
            self.c1 = max(0.6, 0.8 - _ / self.budget)  # Dynamic cognitive parameter
            self.c2 = min(0.9, 1.0 - _ / self.budget)  # Dynamic social parameter
            
            population = self._update_position(population, best_position)
            current_best = population[np.argmin(self._fitness(population, func))]
            best_position = current_best if func(current_best) < func(best_position) else best_position
        
        return best_position
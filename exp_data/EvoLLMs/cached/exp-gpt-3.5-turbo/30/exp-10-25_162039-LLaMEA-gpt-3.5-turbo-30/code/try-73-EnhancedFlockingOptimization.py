import numpy as np

class EnhancedFlockingOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

    def _fitness(self, population, func):
        return np.apply_along_axis(func, 1, population)

    def _update_position(self, population, best_position, w_min=0.4, w_max=1.0, c1=0.8, c2=0.9):
        w = w_max - ((w_max - w_min) * iter_ / self.budget)  # Adaptive inertia weight
        r1, r2 = np.random.rand(self.budget, self.dim), np.random.rand(self.budget, self.dim)
        velocity = w * population + c1 * r1 * (best_position - population) + c2 * r2 * (population - best_position)
        return np.clip(population + velocity, -5.0, 5.0)

    def __call__(self, func):
        population = self._initialize_population()
        best_position = population[np.argmin(self._fitness(population, func))]
        
        for iter_ in range(self.budget):
            population = self._update_position(population, best_position)
            current_best = population[np.argmin(self._fitness(population, func))]
            best_position = current_best if func(current_best) < func(best_position) else best_position
        
        return best_position
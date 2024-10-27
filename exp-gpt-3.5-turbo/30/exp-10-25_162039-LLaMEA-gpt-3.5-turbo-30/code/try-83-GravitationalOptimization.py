import numpy as np

class GravitationalOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

    def _fitness(self, population, func):
        return np.apply_along_axis(func, 1, population)

    def _update_position(self, population, best_position, G=1.0):
        acceleration = G * (best_position - population) / np.linalg.norm(best_position - population, axis=1)[:, None]
        velocity = np.random.rand(self.budget, self.dim) * velocity + acceleration
        return np.clip(population + velocity, -5.0, 5.0)

    def __call__(self, func):
        population = self._initialize_population()
        best_position = population[np.argmin(self._fitness(population, func))]
        
        for _ in range(self.budget):
            population = self._update_position(population, best_position)
            current_best = population[np.argmin(self._fitness(population, func))]
            best_position = current_best if func(current_best) < func(best_position) else best_position
        
        return best_position
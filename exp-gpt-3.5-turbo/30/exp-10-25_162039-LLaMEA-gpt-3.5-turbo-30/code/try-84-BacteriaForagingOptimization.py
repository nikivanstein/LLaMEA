import numpy as np

class BacteriaForagingOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

    def _fitness(self, population, func):
        return np.apply_along_axis(func, 1, population)

    def _update_position(self, population, best_position, chemotaxis_step_size=0.1, elimination_dispersal_step_size=0.2):
        for i in range(self.budget):
            delta = np.random.uniform(-1, 1, size=self.dim)
            delta /= np.linalg.norm(delta)
            population[i] += chemotaxis_step_size * delta
            if func(population[i]) < func(best_position):
                best_position = population[i]
            population[i] += elimination_dispersal_step_size * np.random.uniform(-1, 1, size=self.dim)
        return np.clip(population, -5.0, 5.0)

    def __call__(self, func):
        population = self._initialize_population()
        best_position = population[np.argmin(self._fitness(population, func))]
        
        for _ in range(self.budget):
            population = self._update_position(population, best_position)
        
        return best_position
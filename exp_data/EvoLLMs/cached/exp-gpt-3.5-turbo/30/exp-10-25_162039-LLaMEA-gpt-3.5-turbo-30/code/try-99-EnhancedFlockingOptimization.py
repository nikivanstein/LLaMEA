import numpy as np

class EnhancedFlockingOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

    def _fitness(self, population, func):
        return np.apply_along_axis(func, 1, population)

    def _update_position(self, population, best_position, w_min=0.4, w_max=0.9, c1=1.5, c2=2.0):
        w = w_max - ((w_max - w_min) * (self.budget - 1)) / self.budget
        r1, r2 = np.random.rand(self.budget, self.dim), np.random.rand(self.budget, self.dim)
        velocity = w * population + c1 * r1 * (best_position - population) + c2 * r2 * (population - best_position)
        return np.clip(population + velocity, -5.0, 5.0)

    def _neighborhood_search(self, population, func, neighborhood_size=3):
        for i in range(self.budget):
            indices = np.random.choice(np.delete(np.arange(self.budget), i), neighborhood_size, replace=False)
            trial_population = population[indices]
            current_best = trial_population[np.argmin(self._fitness(trial_population, func))]
            if func(current_best) < func(population[i]):
                population[i] = current_best
        return population

    def __call__(self, func):
        population = self._initialize_population()
        best_position = population[np.argmin(self._fitness(population, func))]
        
        for _ in range(self.budget):
            population = self._update_position(population, best_position)
            population = self._neighborhood_search(population, func)
            current_best = population[np.argmin(self._fitness(population, func))]
            best_position = current_best if func(current_best) < func(best_position) else best_position
        
        return best_position
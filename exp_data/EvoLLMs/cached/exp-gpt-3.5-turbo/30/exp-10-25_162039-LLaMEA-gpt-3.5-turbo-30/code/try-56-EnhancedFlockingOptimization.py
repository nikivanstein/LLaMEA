import numpy as np

class EnhancedFlockingOptimization(FlockingOptimization):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def _update_position(self, population, best_position, w=0.5, c1=0.8, c2=0.9):
        r1, r2 = np.random.rand(self.budget, self.dim), np.random.rand(self.budget, self.dim)
        velocity = w * population + c1 * r1 * (best_position - population) + c2 * r2 * (population - best_position)
        return np.clip(population + velocity, -5.0, 5.0) + np.random.normal(0, 0.01, (self.budget, self.dim))

    def __call__(self, func):
        population = self._initialize_population()
        best_position = population[np.argmin(self._fitness(population, func))]
        
        for _ in range(self.budget):
            population = self._update_position(population, best_position)
            current_best = population[np.argmin(self._fitness(population, func))]
            best_position = current_best if func(current_best) < func(best_position) else best_position
        
        return best_position
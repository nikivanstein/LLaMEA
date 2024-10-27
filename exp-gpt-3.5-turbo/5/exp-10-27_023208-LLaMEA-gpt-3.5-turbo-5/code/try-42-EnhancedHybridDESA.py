import numpy as np

class EnhancedHybridDESA(HybridDESA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.strategy_parameter = 0.5

    def _mutation(self, population, target_index):
        candidates = [idx for idx in range(len(population)) if idx != target_index]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        strategy_factor = 0.5 + 0.1 * np.random.randn(self.dim)
        mutant = population[a] + self.strategy_parameter * strategy_factor * (population[b] - population[c])
        return mutant

    def __call__(self, func):
        population = np.random.uniform(-5, 5, (self.population_size, self.dim))
        return self._optimize_func(func, population)
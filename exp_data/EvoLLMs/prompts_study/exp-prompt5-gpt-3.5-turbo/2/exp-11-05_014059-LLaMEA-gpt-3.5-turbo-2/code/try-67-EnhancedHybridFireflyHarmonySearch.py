import numpy as np

class EnhancedHybridFireflyHarmonySearch(HybridFireflyHarmonySearch):
    def __init__(self, budget, dim, alpha=0.5, beta_min=0.2, beta_max=1.0, harmony_rate=0.5, pitch_adjust_rate=0.5, population_size=20):
        super().__init__(budget, dim, alpha, beta_min, beta_max, harmony_rate, pitch_adjust_rate)
        self.population_size = population_size

    def __call__(self, func):
        def firefly_optimization(population, max_iter):
            # Implementation of firefly optimization with dynamic population size adjustment
            pass

        def harmony_search(population, max_iter):
            # Implementation of harmony search with dynamic population size adjustment
            pass

        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        for _ in range(self.budget // 2):
            population = firefly_optimization(population, max_iter=50)
            population = harmony_search(population, max_iter=50)

        return population[np.argmin([func(ind) for ind in population])]
import numpy as np

class HybridFireflyHarmonySearch:
    def __init__(self, budget, dim, alpha=0.5, beta_min=0.2, beta_max=1.0, harmony_rate=0.5, pitch_adjust_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.harmony_rate = harmony_rate
        self.pitch_adjust_rate = pitch_adjust_rate

    def __call__(self, func):
        def firefly_optimization(population, max_iter):
            # Implementation of firefly optimization
            pass

        def harmony_search(population, max_iter):
            # Implementation of harmony search with adaptive population size adjustment
            pass

        population = np.random.uniform(-5.0, 5.0, (self.dim, self.dim))
        for _ in range(self.budget // 2):
            population = firefly_optimization(population, max_iter=50)
            population = harmony_search(population, max_iter=50)

        return population[np.argmin([func(ind) for ind in population])]
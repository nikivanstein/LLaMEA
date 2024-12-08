import numpy as np

class EnhancedAdaptiveDE(AdaptiveDE):
    def __init__(self, budget, dim, F=0.8, CR=0.9, pop_size=20, F_scale=1.0):
        super().__init__(budget, dim, F, CR, pop_size)
        self.F_scale = F_scale

    def __call__(self, func):
        def mutate(x, population, F):
            a, b, c = population[np.random.choice(len(population), 3, replace=False)]
            return np.clip(a + F * self.F_scale * (b - c), -5, 5)
        
        # Rest of the code remains the same

        return population[best_idx]
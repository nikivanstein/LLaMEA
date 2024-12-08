import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.max_iter = budget // self.pop_size
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, dim))
        self.population[0] = np.random.uniform(-5.0, 5.0, dim)  # Chaotic initialization for diversity
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.cr = 0.9
        self.f = 0.8
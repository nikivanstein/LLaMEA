import numpy as np

class EnhancedEDHS(EDHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.hmcr = np.random.uniform(0.7, 0.9)
        self.alpha = np.random.uniform(0.7, 0.9)
        self.mutation_rate = np.random.uniform(0.2, 0.4)

    def __call__(self, func):
        return super().__call__(func)
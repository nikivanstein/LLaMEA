import numpy as np

class EnhancedAdaptiveDE(ImprovedAdaptiveDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.F_min, self.F_max = 0.3, 0.7
        self.CR_min, self.CR_max = 0.4, 0.6

    def mutate(self, pop, best, F):
        # Improved mutation strategy implementation
        ...

    def crossover(self, target, mutant, CR):
        # Improved crossover strategy implementation
        ...

    def __call__(self, func):
        return super().__call__(func)
import numpy as np

class EnhancedAdaptiveDE(ImprovedAdaptiveDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.F_min, self.F_max = 0.25, 0.75
        self.CR_min, self.CR_max = 0.3, 0.7

    def mutate(self, pop, best, F):
        # Further improved mutation strategy implementation
        ...

    def crossover(self, target, mutant, CR):
        # Further improved crossover strategy implementation
        ...

    def __call__(self, func):
        return super().__call__(func)
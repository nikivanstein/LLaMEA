import numpy as np

class EnhancedOppositionBasedDE(ImprovedAdaptiveDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.F_min, self.F_max = 0.3, 0.7
        self.F_op_min, self.F_op_max = 0.1, 0.9
        self.CR_min, self.CR_max = 0.4, 0.6
        self.CR_op_min, self.CR_op_max = 0.2, 0.8

    def mutate(self, pop, best, F):
        # Enhanced mutation strategy implementation with opposition-based learning
        ...

    def crossover(self, target, mutant, CR):
        # Enhanced crossover strategy implementation with opposition-based learning
        ...

    def __call__(self, func):
        return super().__call__(func)
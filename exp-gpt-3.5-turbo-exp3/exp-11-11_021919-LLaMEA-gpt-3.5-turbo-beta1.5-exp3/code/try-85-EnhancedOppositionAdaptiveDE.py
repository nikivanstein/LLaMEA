import numpy as np

class EnhancedOppositionAdaptiveDE(ImprovedAdaptiveDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.F_min, self.F_max = 0.4, 0.9
        self.CR_min, self.CR_max = 0.3, 0.7

    def initialize_population(self, bounds):
        # Novel opposition-based initialization strategy implementation
        ...

    def select_parents(self, pop, current):
        # Probabilistic selection of parents strategy implementation
        ...

    def __call__(self, func):
        return super().__call__(func)
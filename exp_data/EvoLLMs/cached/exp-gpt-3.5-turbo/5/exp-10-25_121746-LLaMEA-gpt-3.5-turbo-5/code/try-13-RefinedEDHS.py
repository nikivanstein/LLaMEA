import numpy as np

class RefinedEDHS(EDHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.hmcr = 0.8
        self.alpha = 0.8
        self.mutation_rate = 0.3

    def __call__(self, func):
        # Code remains the same as EDHS, with updated parameters for individual refinement
        return super().__call__(func)
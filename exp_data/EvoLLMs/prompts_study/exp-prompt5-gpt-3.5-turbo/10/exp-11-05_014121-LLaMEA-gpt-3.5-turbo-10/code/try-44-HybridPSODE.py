import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        def pso_de_optimizer():
            # PSO here
            pass
        
        return pso_de_optimizer()
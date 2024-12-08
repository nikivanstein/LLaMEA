import numpy as np

class Refined_Adaptive_DE_PSO_Metaheuristic(Adaptive_DE_PSO_Metaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.pop_size = 50

    def __call__(self, func):
        super_call = super().__call__(func)
        return super_call
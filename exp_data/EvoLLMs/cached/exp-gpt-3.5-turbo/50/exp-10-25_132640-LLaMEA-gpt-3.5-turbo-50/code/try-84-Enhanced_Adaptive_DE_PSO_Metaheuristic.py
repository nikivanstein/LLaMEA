import numpy as np

class Enhanced_Adaptive_DE_PSO_Metaheuristic(Adaptive_DE_PSO_Metaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.F_min = 0.5
        self.F_max = 0.9

    def __call__(self, func):
        self.c1_min = 1.7
        self.c1_max = 2.0

        return super().__call__(func)
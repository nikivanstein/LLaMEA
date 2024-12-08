import numpy as np

class Enhanced_Adaptive_DE_PSO_Metaheuristic_LineRefinement(Adaptive_DE_PSO_Metaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.F_min = 0.5
        self.F_max = 0.9

    def __call__(self, func):
        self.c1_min = 1.7
        self.c1_max = 2.0
        
        # Implement line refinement strategy with probability 0.5
        if np.random.rand() < 0.5:
            # Refine individual lines of the selected solution

        return super().__call__(func)
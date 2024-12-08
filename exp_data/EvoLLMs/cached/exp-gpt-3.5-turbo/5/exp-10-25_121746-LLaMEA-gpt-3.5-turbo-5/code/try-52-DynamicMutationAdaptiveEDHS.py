import numpy as np

class DynamicMutationAdaptiveEDHS(EnhancedEDHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = np.random.uniform(0.2, 0.4)  

    def __call__(self, func):
        for t in range(1, self.budget+1):
            self.mutation_rate = 0.2 + 0.2 * np.cos(t / self.budget * np.pi)  # Dynamic mutation rate update
            super().__call__(func)
        return self.get_global_best()
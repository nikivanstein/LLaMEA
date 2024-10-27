import numpy as np

class VariableNeighborhoodProbabilisticAdaptiveEDHS(EnhancedEDHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = np.random.uniform(0.2, 0.4)  # Resetting mutation rate

    def __call__(self, func):
        for _ in range(self.budget):
            # Update mutation rate based on a time-varying strategy with a probability
            if np.random.rand() < 0.05:
                self.mutation_rate = np.random.uniform(0.1, 0.5)
            # Integrate variable neighborhood search to explore diverse regions
            if np.random.rand() < 0.1:
                self.current_solution = self.explore_neighborhood(self.current_solution)
            super().__call__(func)
        return self.get_global_best()
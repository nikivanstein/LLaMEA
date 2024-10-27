import numpy as np

class DynamicEDHS(EnhancedEDHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = np.random.uniform(0.2, 0.4)  # Resetting mutation rate

    def __call__(self, func):
        for _ in range(self.budget):
            # Update mutation rate based on a dynamic time-varying strategy
            self.mutation_rate = self.mutation_rate * np.exp(-0.01 * np.random.randint(1, 5))
            super().__call__(func)
        return self.get_global_best()
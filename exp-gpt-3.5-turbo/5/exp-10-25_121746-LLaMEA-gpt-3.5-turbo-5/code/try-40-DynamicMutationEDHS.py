import numpy as np

class DynamicMutationEDHS(EnhancedEDHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = np.random.uniform(0.2, 0.4)  # Resetting mutation rate

    def __call__(self, func):
        history = []
        for _ in range(self.budget):
            # Update mutation rate based on population evolution
            history.append(super().__call__(func))
            if len(history) > 1 and history[-1] < history[-2]:
                self.mutation_rate *= 1.1  # Increase mutation rate
            else:
                self.mutation_rate *= 0.9  # Decrease mutation rate
        return self.get_global_best()
import numpy as np

class ImprovedParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        inertia_weight = 0.5 + 0.3 * np.cos(2 * np.pi * np.arange(1, self.budget + 1) / self.budget)  # Dynamic inertia weight
        # PSO optimization implementation with dynamic inertia weight
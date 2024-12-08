import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w = 0.5  # Inertia weight
        self.c1 = 2.0  # Cognitive factor
        self.c2 = 2.0  # Social factor

    def __call__(self, func):
        # PSO optimization implementation with dynamic inertia weight and personal best learning
        pass
class ImprovedParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.inertia_weight = 0.5  # Initial inertia weight
        self.constriction_factor = 0.8  # Constriction factor

    def __call__(self, func):
        # Improved PSO optimization implementation
        pass
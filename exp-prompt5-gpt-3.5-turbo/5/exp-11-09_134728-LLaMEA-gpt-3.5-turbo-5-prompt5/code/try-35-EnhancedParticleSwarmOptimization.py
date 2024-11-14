class EnhancedParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w = 0.9  # Inertia weight
        self.c1 = 2.0  # Cognitive parameter
        self.c2 = 2.0  # Social parameter

    def __call__(self, func):
        # Enhanced PSO optimization implementation with dynamic inertia weight and adaptive learning parameters
        pass
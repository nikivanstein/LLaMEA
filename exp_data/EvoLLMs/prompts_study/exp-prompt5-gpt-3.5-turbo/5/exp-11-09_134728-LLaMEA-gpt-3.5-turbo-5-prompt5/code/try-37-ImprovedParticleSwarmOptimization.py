class ImprovedParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.inertia_weight = 0.5
        self.cognitive_weight = 0.8
        self.social_weight = 0.6

    def __call__(self, func):
        # Improved PSO optimization with dynamic parameter adaptation
        pass
import numpy as np

class EnhancedPSOSimulatedAnnealing(PSOSimulatedAnnealing):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.inertia_min = 0.1
        self.inertia_max = 1.0

    def __call__(self, func):
        def update_position(particle, pbest):
            inertia_weight = self.inertia_min + (self.inertia_max - self.inertia_min) * np.random.rand()
            new_velocity = inertia_weight * particle['velocity'] + self.cognitive_weight * np.random.rand() * (pbest['position'] - particle['position']) + self.social_weight * np.random.rand() * (gbest['position'] - particle['position'])
            new_position = particle['position'] + new_velocity
            return new_position

        return super().__call__(func)
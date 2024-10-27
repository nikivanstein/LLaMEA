import numpy as np

class RefinedPSOSimulatedAnnealing(PSOSimulatedAnnealing):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.dynamic_inertia_weight = True

    def __call__(self, func):
        def update_position(particle, pbest):
            inertia_weight = self.inertia_weight if not self.dynamic_inertia_weight else np.clip(0.9 - 0.8 * _ / self.budget, 0.1, 0.9)
            velocity = particle['velocity']
            position = particle['position']
            new_velocity = inertia_weight * velocity + self.cognitive_weight * np.random.rand() * (pbest['position'] - position) + self.social_weight * np.random.rand() * (gbest['position'] - position)
            new_position = np.clip(position + new_velocity, -3.0, 3.0)
            return new_position
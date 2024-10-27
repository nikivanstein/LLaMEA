# import numpy as np

class DynamicInertiaPSOSimulatedAnnealing(PSOSimulatedAnnealing):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.min_inertia = 0.1
        self.max_inertia = 0.9

    def __call__(self, func):
        def update_position(particle, pbest):
            inertia_weight = self.min_inertia + (self.max_inertia - self.min_inertia) * (1 - iteration / self.budget)
            new_velocity = inertia_weight * particle['velocity'] + self.cognitive_weight * np.random.rand() * (pbest['position'] - particle['position']) + self.social_weight * np.random.rand() * (gbest['position'] - particle['position'])
            new_position = particle['position'] + new_velocity
            return new_position

        # Rest of the code remains the same
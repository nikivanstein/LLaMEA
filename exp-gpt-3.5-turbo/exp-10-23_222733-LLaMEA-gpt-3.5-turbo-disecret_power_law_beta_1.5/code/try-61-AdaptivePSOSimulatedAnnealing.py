import numpy as np

class AdaptivePSOSimulatedAnnealing(PSOSimulatedAnnealing):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.alpha = (self.initial_temp - self.final_temp) / budget
        self.current_temp = self.initial_temp

    def __call__(self, func):
        def update_position(particle, pbest):
            # Enhanced position update mechanism
            inertia_weight = self.inertia_weight * np.exp(-0.01 * self.budget)
            cognitive_weight = self.cognitive_weight * np.exp(-0.01 * self.budget)
            social_weight = self.social_weight * np.exp(-0.01 * self.budget)

            velocity = particle['velocity']
            position = particle['position']
            new_velocity = inertia_weight * velocity + cognitive_weight * np.random.rand() * (pbest['position'] - position) + social_weight * np.random.rand() * (gbest['position'] - position)
            new_position = position + new_velocity
            return new_position

        return super().__call__(func)
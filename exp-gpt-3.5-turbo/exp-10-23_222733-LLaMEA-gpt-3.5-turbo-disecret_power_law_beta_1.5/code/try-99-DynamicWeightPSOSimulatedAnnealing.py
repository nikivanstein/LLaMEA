import numpy as np

class DynamicWeightPSOSimulatedAnnealing(PSOSimulatedAnnealing):
    def __call__(self, func):
        def update_position(particle, pbest):
            inertia_weight = self.inertia_weight * np.random.uniform(0.8, 1.2)  # Adjust inertia weight dynamically
            cognitive_weight = self.cognitive_weight * np.random.uniform(0.8, 1.2)  # Adjust cognitive weight dynamically
            social_weight = self.social_weight * np.random.uniform(0.8, 1.2)  # Adjust social weight dynamically
            new_velocity = inertia_weight * particle['velocity'] + cognitive_weight * np.random.rand() * (pbest['position'] - particle['position']) + social_weight * np.random.rand() * (gbest['position'] - particle['position'])
            new_position = particle['position'] + new_velocity
            return new_position

        return super().__call__(func)
# import numpy as np

class DynamicInertiaPSOSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 20
        self.initial_inertia_weight = 0.9
        self.final_inertia_weight = 0.4
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.initial_temp = 1.0
        self.final_temp = 0.001
        self.alpha = (self.initial_temp - self.final_temp) / budget
        self.current_temp = self.initial_temp
    
    def __call__(self, func):
        def update_position(particle, pbest):
            inertia_weight = self.initial_inertia_weight - ((self.initial_inertia_weight - self.final_inertia_weight) * iter_count / self.budget)
            velocity = particle['velocity']
            position = particle['position']
            new_velocity = inertia_weight * velocity + self.cognitive_weight * np.random.rand() * (pbest['position'] - position) + self.social_weight * np.random.rand() * (gbest['position'] - position)
            new_position = position + new_velocity
            return new_position

        # Rest of the code remains the same
        
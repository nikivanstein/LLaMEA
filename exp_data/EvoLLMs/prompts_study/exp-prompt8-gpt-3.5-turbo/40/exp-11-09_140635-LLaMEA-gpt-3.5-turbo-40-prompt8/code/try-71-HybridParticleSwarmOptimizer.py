import numpy as np

class HybridParticleSwarmOptimizer:
    def __init__(self, budget, dim, num_particles=10, inertia_weight=0.5, cognitive_param=1.5, social_param=2.0):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_param = cognitive_param
        self.social_param = social_param

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        best_position = swarm[np.argmin([func(p) for p in swarm])]
        
        for _ in range(self.budget):
            for i in range(self.num_particles):
                cognitive_velocity = self.cognitive_param * np.random.random() * (best_position - swarm[i])
                social_velocity = self.social_param * np.random.random() * (swarm[np.argmin([func(p) for p in swarm])] - swarm[i])
                velocities[i] = self.inertia_weight * velocities[i] + cognitive_velocity + social_velocity
                swarm[i] = np.clip(swarm[i] + velocities[i], -5.0, 5.0)
                if func(swarm[i]) < func(best_position):
                    best_position = swarm[i]
        return best_position
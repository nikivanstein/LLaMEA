import numpy as np

class ParticleSwarmOptimizationMetaheuristic:
    def __init__(self, budget, dim, inertia_weight=0.5, cognitive_weight=1.5, social_weight=2.0, vmax=0.5):
        self.budget = budget
        self.dim = dim
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.vmax = vmax

    def __call__(self, func):
        swarm_position = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        swarm_velocity = np.random.uniform(-self.vmax, self.vmax, (self.budget, self.dim))
        personal_best = swarm_position.copy()
        global_best = swarm_position[np.argmin([func(p) for p in swarm_position])]

        for _ in range(self.budget):
            for i in range(self.budget):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                swarm_velocity[i] = self.inertia_weight * swarm_velocity[i] + \
                                     self.cognitive_weight * r1 * (personal_best[i] - swarm_position[i]) + \
                                     self.social_weight * r2 * (global_best - swarm_position[i])
                swarm_velocity[i] = np.clip(swarm_velocity[i], -self.vmax, self.vmax)
                swarm_position[i] += swarm_velocity[i]
                
                if func(swarm_position[i]) < func(personal_best[i]):
                    personal_best[i] = swarm_position[i]
                    if func(swarm_position[i]) < func(global_best):
                        global_best = swarm_position[i]

        return global_best
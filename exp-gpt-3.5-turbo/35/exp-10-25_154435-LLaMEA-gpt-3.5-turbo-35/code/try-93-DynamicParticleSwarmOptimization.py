import numpy as np

class DynamicParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.lb = -5.0
        self.ub = 5.0
        self.inertia_weight = 0.5
        self.min_cognitive_weight = 0.5
        self.max_cognitive_weight = 2.5
        self.min_social_weight = 0.5
        self.max_social_weight = 2.5
        self.max_velocity = 0.1 * (self.ub - self.lb)

    def __call__(self, func):
        def initialize_swarm(swarm_size, dim, lb, ub):
            positions = np.random.uniform(lb, ub, (swarm_size, dim))
            velocities = np.random.uniform(-0.1, 0.1, (swarm_size, dim))
            return positions, velocities

        def update_velocity(position, velocity, p_best, g_best, cognitive_weight, social_weight):
            inertia_term = self.inertia_weight * velocity
            cognitive_term = cognitive_weight * np.random.rand() * (p_best - position)
            social_term = social_weight * np.random.rand() * (g_best - position)
            new_velocity = inertia_term + cognitive_term + social_term
            return np.clip(new_velocity, -self.max_velocity, self.max_velocity)

        def update_position(position, velocity, lb, ub):
            new_position = position + velocity
            return np.clip(new_position, lb, ub)

        swarm, velocities = initialize_swarm(self.swarm_size, self.dim, self.lb, self.ub)
        p_best = swarm.copy()
        g_best = p_best[np.argmin([func(p) for p in p_best])
        for _ in range(self.budget // self.swarm_size):
            cognitive_weight = np.interp(_, [0, self.budget // self.swarm_size], [self.min_cognitive_weight, self.max_cognitive_weight])
            social_weight = np.interp(_, [0, self.budget // self.swarm_size], [self.min_social_weight, self.max_social_weight])
            for i in range(self.swarm_size):
                velocities[i] = update_velocity(swarm[i], velocities[i], p_best[i], g_best, cognitive_weight, social_weight)
                swarm[i] = update_position(swarm[i], velocities[i], self.lb, self.ub)
                if func(swarm[i]) < func(p_best[i]):
                    p_best[i] = swarm[i]
            g_best = p_best[np.argmin([func(p) for p in p_best])]
        return g_best
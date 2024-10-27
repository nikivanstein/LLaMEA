import numpy as np

class DynamicInertiaPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.lb = -5.0
        self.ub = 5.0
        self.inertia_min = 0.4
        self.inertia_max = 0.9
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.max_velocity = 0.1 * (self.ub - self.lb)

    def __call__(self, func):
        def initialize_swarm(swarm_size, dim, lb, ub):
            positions = np.random.uniform(lb, ub, (swarm_size, dim))
            velocities = np.random.uniform(-0.1, 0.1, (swarm_size, dim))
            return positions, velocities

        def update_velocity(position, velocity, p_best, g_best, inertia_weight):
            inertia_term = inertia_weight * velocity
            cognitive_term = self.cognitive_weight * np.random.rand() * (p_best - position)
            social_term = self.social_weight * np.random.rand() * (g_best - position)
            new_velocity = inertia_term + cognitive_term + social_term
            return np.clip(new_velocity, -self.max_velocity, self.max_velocity)

        def update_position(position, velocity, lb, ub):
            new_position = position + velocity
            return np.clip(new_position, lb, ub)

        swarm, velocities = initialize_swarm(self.swarm_size, self.dim, self.lb, self.ub)
        p_best = swarm.copy()
        g_best = p_best[np.argmin([func(p) for p in p_best])
        inertia_weight = self.inertia_max
        for _ in range(self.budget // self.swarm_size):
            for i in range(self.swarm_size):
                velocities[i] = update_velocity(swarm[i], velocities[i], p_best[i], g_best, inertia_weight)
                swarm[i] = update_position(swarm[i], velocities[i], self.lb, self.ub)
                if func(swarm[i]) < func(p_best[i]):
                    p_best[i] = swarm[i]
            g_best = p_best[np.argmin([func(p) for p in p_best])]
            inertia_weight = self.inertia_min + (_ / (self.budget // self.swarm_size)) * (self.inertia_max - self.inertia_min)
        return g_best
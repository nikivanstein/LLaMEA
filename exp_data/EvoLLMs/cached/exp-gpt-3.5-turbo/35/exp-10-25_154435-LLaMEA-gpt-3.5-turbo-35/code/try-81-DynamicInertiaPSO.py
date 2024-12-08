import numpy as np

class DynamicInertiaPSO(ParticleSwarmOptimization):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.inertia_min = 0.4
        self.inertia_max = 0.9

    def __call__(self, func):
        def update_velocity(position, velocity, p_best, g_best):
            inertia_weight = self.inertia_min + (self.inertia_max - self.inertia_min) * (self.budget - i) / self.budget
            inertia_term = inertia_weight * velocity
            cognitive_term = self.cognitive_weight * np.random.rand() * (p_best - position)
            social_term = self.social_weight * np.random.rand() * (g_best - position)
            new_velocity = inertia_term + cognitive_term + social_term
            return np.clip(new_velocity, -self.max_velocity, self.max_velocity)

        swarm, velocities = initialize_swarm(self.swarm_size, self.dim, self.lb, self.ub)
        p_best = swarm.copy()
        g_best = p_best[np.argmin([func(p) for p in p_best])
        for i in range(self.budget):
            for i in range(self.swarm_size):
                velocities[i] = update_velocity(swarm[i], velocities[i], p_best[i], g_best)
                swarm[i] = update_position(swarm[i], velocities[i], self.lb, self.ub)
                if func(swarm[i]) < func(p_best[i]):
                    p_best[i] = swarm[i]
            g_best = p_best[np.argmin([func(p) for p in p_best])]
        return g_best
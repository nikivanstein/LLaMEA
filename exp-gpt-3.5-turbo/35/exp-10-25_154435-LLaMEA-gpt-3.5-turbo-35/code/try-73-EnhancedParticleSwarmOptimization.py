import numpy as np

class EnhancedParticleSwarmOptimization(ParticleSwarmOptimization):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.inertia_decay = 0.95

    def __call__(self, func):
        def adaptive_update_velocity(position, velocity, p_best, g_best):
            inertia_term = self.inertia_weight * velocity
            cognitive_term = self.cognitive_weight * np.random.rand() * (p_best - position)
            social_term = self.social_weight * np.random.rand() * (g_best - position)
            new_velocity = self.inertia_weight * inertia_term + cognitive_term + social_term
            return np.clip(new_velocity, -self.max_velocity, self.max_velocity)

        swarm, velocities = self.initialize_swarm(self.swarm_size, self.dim, self.lb, self.ub)
        p_best = swarm.copy()
        g_best = p_best[np.argmin([func(p) for p in p_best])]
        for _ in range(self.budget // self.swarm_size):
            for i in range(self.swarm_size):
                velocities[i] = self.adaptive_update_velocity(swarm[i], velocities[i], p_best[i], g_best)
                swarm[i] = self.update_position(swarm[i], velocities[i], self.lb, self.ub)
                if func(swarm[i]) < func(p_best[i]):
                    p_best[i] = swarm[i]
            g_best = p_best[np.argmin([func(p) for p in p_best])]
            self.inertia_weight *= self.inertia_decay
        return g_best
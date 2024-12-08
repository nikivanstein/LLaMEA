import numpy as np

class DynamicParameterAdaptationPSO(ParticleSwarmOptimization):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.min_inertia_weight = 0.4
        self.max_inertia_weight = 0.9
        self.min_cognitive_weight = 0.5
        self.max_cognitive_weight = 2.0
        self.min_social_weight = 0.5
        self.max_social_weight = 2.0

    def __call__(self, func):
        def update_parameters(iteration):
            self.inertia_weight = self.min_inertia_weight + (self.max_inertia_weight - self.min_inertia_weight) * (iteration / (self.budget // self.swarm_size))
            self.cognitive_weight = self.min_cognitive_weight + (self.max_cognitive_weight - self.min_cognitive_weight) * (iteration / (self.budget // self.swarm_size))
            self.social_weight = self.min_social_weight + (self.max_social_weight - self.min_social_weight) * (iteration / (self.budget // self.swarm_size))

        swarm, velocities = initialize_swarm(self.swarm_size, self.dim, self.lb, self.ub)
        p_best = swarm.copy()
        g_best = p_best[np.argmin([func(p) for p in p_best])]
        for iteration in range(self.budget // self.swarm_size):
            update_parameters(iteration)
            for i in range(self.swarm_size):
                velocities[i] = update_velocity(swarm[i], velocities[i], p_best[i], g_best)
                swarm[i] = update_position(swarm[i], velocities[i], self.lb, self.ub)
                if func(swarm[i]) < func(p_best[i]):
                    p_best[i] = swarm[i]
            g_best = p_best[np.argmin([func(p) for p in p_best])]
        return g_best
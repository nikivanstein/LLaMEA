import numpy as np

class AdaptiveMultiPopEnhancedHybridPSOLS(MultiPopEnhancedHybridPSOLS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.phi1_min = 0.1
        self.phi1_max = 0.9
        self.phi2_min = 0.1
        self.phi2_max = 0.9

    def __call__(self, func):
        def adaptive_pso(swarm_pos):
            swarm_val = np.array([func(pos) for pos in swarm_pos])
            pbest_pos = swarm_pos.copy()
            pbest_val = swarm_val.copy()
            gbest_idx = np.argmin(swarm_val)
            gbest_pos = swarm_pos[gbest_idx].copy()
            gbest_val = swarm_val[gbest_idx]

            for _ in range(self.max_iter):
                for i in range(self.num_particles):
                    phi1 = self.phi1_min + (self.phi1_max - self.phi1_min) * np.random.rand()
                    phi2 = self.phi2_min + (self.phi2_max - self.phi2_min) * np.random.rand()
                    omega = 0.5 + 0.2 * np.random.rand()  # Dynamic inertia weight
                    new_pos = swarm_pos[i] + phi1 * (pbest_pos[i] - swarm_pos[i]) + phi2 * (gbest_pos - swarm_pos[i])
                    new_pos = np.clip(new_pos, -5.0, 5.0)
                    new_val = func(new_pos)
                    if new_val < swarm_val[i]:
                        swarm_pos[i] = new_pos
                        swarm_val[i] = new_val
                        if new_val < pbest_val[i]:
                            pbest_pos[i] = new_pos
                            pbest_val[i] = new_val
                            if new_val < gbest_val:
                                gbest_pos = new_pos
                                gbest_val = new_val

            return gbest_pos, gbest_val

        best_pos = np.random.uniform(-5.0, 5.0, self.dim)
        best_val = func(best_pos)
        
        for _ in range(self.budget // (self.num_particles * self.num_populations * 12)):
            swarm_pos = [np.random.uniform(-5.0, 5.0, (self.num_particles, self.dim)) for _ in range(self.num_populations)]
            for i in range(self.num_populations):
                new_pos, new_val = adaptive_pso(swarm_pos[i])
                new_pos, new_val = local_search(new_pos, new_val)
                if new_val < best_val:
                    best_pos = new_pos
                    best_val = new_val

        return best_pos
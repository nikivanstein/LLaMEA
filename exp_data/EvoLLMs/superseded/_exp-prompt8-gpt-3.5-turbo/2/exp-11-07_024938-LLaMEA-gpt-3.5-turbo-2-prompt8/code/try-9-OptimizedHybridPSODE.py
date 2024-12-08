import numpy as np

class OptimizedHybridPSODE(HybridPSODE):
    def __call__(self, func):
        swarm = self.initialize_particles()
        velocities = np.zeros((self.num_particles, self.dim))
        pbest = swarm.copy()
        pbest_vals = np.apply_along_axis(func, 1, pbest)
        gbest_idx = np.argmin(pbest_vals)
        gbest = pbest[gbest_idx]
        gbest_val = pbest_vals[gbest_idx]

        for _ in range(self.budget):
            r1_r2 = np.random.rand(2, self.num_particles, self.dim)
            velocities = self.w * velocities + self.c1 * r1_r2[0] * (pbest - swarm) + self.c2 * r1_r2[1] * (gbest - swarm)
            swarm = np.clip(swarm + velocities, self.bounds[0], self.bounds[1])

            candidates = np.array([self.mutate(swarm[np.random.choice(self.num_particles)], gbest, swarm[i]) for i in range(self.num_particles)])
            candidate_vals = np.apply_along_axis(func, 1, candidates)

            update_indices = candidate_vals < pbest_vals
            pbest[update_indices] = candidates[update_indices]
            pbest_vals[update_indices] = candidate_vals[update_indices]

            new_gbest_idx = np.argmin(pbest_vals)
            if pbest_vals[new_gbest_idx] < gbest_val:
                gbest = pbest[new_gbest_idx]
                gbest_val = pbest_vals[new_gbest_idx]

        return gbest
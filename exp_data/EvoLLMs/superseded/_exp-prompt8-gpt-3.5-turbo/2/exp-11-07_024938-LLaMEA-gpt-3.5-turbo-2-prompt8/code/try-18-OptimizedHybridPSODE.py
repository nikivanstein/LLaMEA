import numpy as np

class OptimizedHybridPSODE(HybridPSODE):
    def __call__(self, func):
        swarm = self.initialize_particles()
        velocities = np.zeros((self.num_particles, self.dim))
        pbest = swarm.copy()
        pbest_vals = np.apply_along_axis(func, 1, pbest)
        gbest = pbest[np.argmin(pbest_vals)]
        gbest_val = np.min(pbest_vals)
        
        for _ in range(self.budget):
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = self.w * velocities[i] + self.c1 * r1 * (pbest[i] - swarm[i]) + self.c2 * r2 * (gbest - swarm[i])
                swarm[i] = np.clip(swarm[i] + velocities[i], self.bounds[0], self.bounds[1])
                
                candidate = self.mutate(swarm[np.random.choice(self.num_particles)], gbest, swarm[i])
                candidate_val = func(candidate)
                
                if candidate_val < pbest_vals[i]:
                    pbest[i] = candidate
                    pbest_vals[i] = candidate_val
                    if candidate_val < gbest_val:
                        gbest = candidate
                        gbest_val = candidate_val
        
        return gbest
import numpy as np

class MOPSO_LS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 25
        self.max_iter = budget // self.num_particles

    def __call__(self, func):
        def local_search(current_pos, current_val):
            epsilon = 1e-5
            best_pos = current_pos
            best_val = current_val
            for _ in range(15):
                new_pos = best_pos + (np.random.rand(self.dim) - 0.5) * epsilon
                new_pos = np.clip(new_pos, -5.0, 5.0)
                new_val = func(new_pos)
                if new_val < best_val:
                    best_pos = new_pos
                    best_val = new_val
            return best_pos, best_val

        def pso(func):
            swarm_pos = np.random.uniform(-5.0, 5.0, (self.num_particles, self.dim))
            swarm_val = np.array([func(pos) for pos in swarm_pos])
            pbest_pos = swarm_pos.copy()
            pbest_val = swarm_val.copy()
            gbest_idx = np.argmin(swarm_val)
            gbest_pos = swarm_pos[gbest_idx].copy()
            gbest_val = swarm_val[gbest_idx]

            for _ in range(self.max_iter):
                for i in range(self.num_particles):
                    phi1 = 0.6 + 0.1 * np.random.rand()
                    phi2 = 0.4 + 0.1 * np.random.rand()
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
        
        for _ in range(self.budget // (self.num_particles * 12)):
            new_pos, new_val = pso(func)
            new_pos, new_val = local_search(new_pos, new_val)
            if new_val < best_val:
                best_pos = new_pos
                best_val = new_val

        return best_pos
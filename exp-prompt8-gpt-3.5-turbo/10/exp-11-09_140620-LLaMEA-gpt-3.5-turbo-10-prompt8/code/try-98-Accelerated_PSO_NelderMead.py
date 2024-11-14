class Accelerated_PSO_NelderMead:
    def __init__(self, budget, dim, swarm_size=30, max_iter=100, inertia_min=0.4, inertia_max=0.9, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.lb = -5.0
        self.ub = 5.0
        self.inertia_min = inertia_min
        self.inertia_max = inertia_max
        self.c1 = c1
        self.c2 = c2

    def __call__(self, func):
        swarm = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        pbest = swarm.copy()
        velocity = np.zeros((self.swarm_size, self.dim))
        gbest_idx = np.argmin([func(p) for p in swarm])
        gbest = swarm[gbest_idx].copy()
        
        inertia = self.inertia_max

        for _ in range(self.max_iter):
            for i in range(self.swarm_size):
                new_velocity = inertia * velocity[i] + self.c1 * np.random.rand() * (pbest[i] - swarm[i]) + self.c2 * np.random.rand() * (gbest - swarm[i])
                new_position = swarm[i] + new_velocity
                new_position = np.clip(new_position, self.lb, self.ub)

                if func(new_position) < func(pbest[i]):
                    pbest[i] = new_position.copy()

                swarm[i] = new_position.copy()

                if func(new_position) < func(gbest):
                    gbest = new_position.copy()
            
            simplex = [gbest + np.random.normal(0, 0.5, self.dim) for _ in range(self.dim + 1)]
            gbest = self.optimize_simplex(simplex, func)

            # Adapt inertia dynamically based on convergence speed
            if np.random.rand() < 0.1:
                prev_gbest = gbest
                gbest_idx = np.argmin([func(p) for p in swarm])
                gbest = swarm[gbest_idx].copy()
                if func(gbest) >= func(prev_gbest):
                    inertia = max(self.inertia_min, inertia - 0.05)
                else:
                    inertia = min(self.inertia_max, inertia + 0.05)

        return gbest
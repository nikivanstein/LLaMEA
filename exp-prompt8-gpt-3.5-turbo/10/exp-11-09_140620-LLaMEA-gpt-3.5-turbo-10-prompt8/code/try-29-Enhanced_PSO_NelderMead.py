class Enhanced_PSO_NelderMead:
    def __init__(self, budget, dim, swarm_size=30, max_iter=100):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.lb = -5.0
        self.ub = 5.0
        self.w_max = 0.9
        self.w_min = 0.4

    def __call__(self, func):
        swarm = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        pbest = swarm.copy()
        velocity = np.zeros((self.swarm_size, self.dim))
        gbest_idx = np.argmin([func(p) for p in swarm])
        gbest = swarm[gbest_idx].copy()
        w = self.w_max

        for _ in range(self.max_iter):
            for i in range(self.swarm_size):
                new_velocity = w * velocity[i] + np.random.rand() * (pbest[i] - swarm[i]) + np.random.rand() * (gbest - swarm[i])
                new_position = swarm[i] + new_velocity
                new_position = np.clip(new_position, self.lb, self.ub)

                if func(new_position) < func(pbest[i]):
                    pbest[i] = new_position.copy()

                swarm[i] = new_position.copy()

                if func(new_position) < func(gbest):
                    gbest = new_position.copy()

            simplex = [gbest + np.random.normal(0, 0.5, self.dim) for _ in range(self.dim + 1)]
            gbest = self.optimize_simplex(simplex, func)

            improvement_rate = 1.0 - (func(gbest) / func(swarm[gbest_idx]))
            w = self.w_max - (self.w_max - self.w_min) * improvement_rate

        return gbest
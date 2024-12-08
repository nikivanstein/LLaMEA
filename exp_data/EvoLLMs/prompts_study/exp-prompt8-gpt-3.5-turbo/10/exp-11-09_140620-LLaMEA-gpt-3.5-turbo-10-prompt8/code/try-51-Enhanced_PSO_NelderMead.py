class Enhanced_PSO_NelderMead:
    def __init__(self, budget, dim, swarm_size=30, max_iter=100, inertia_min=0.4, inertia_max=0.9, w_change_iter=10):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.lb = -5.0
        self.ub = 5.0
        self.inertia_min = inertia_min
        self.inertia_max = inertia_max
        self.w_change_iter = w_change_iter

    def __call__(self, func):
        swarm = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        pbest = swarm.copy()
        velocity = np.zeros((self.swarm_size, self.dim))
        gbest_idx = np.argmin([func(p) for p in swarm])
        gbest = swarm[gbest_idx].copy()
        inertial_weight = self.inertia_max

        for it in range(self.max_iter):
            for i in range(self.swarm_size):
                new_velocity = inertial_weight * velocity[i] + np.random.rand() * (pbest[i] - swarm[i]) + np.random.rand() * (gbest - swarm[i])
                new_position = swarm[i] + new_velocity
                new_position = np.clip(new_position, self.lb, self.ub)

                if func(new_position) < func(pbest[i]):
                    pbest[i] = new_position.copy()

                swarm[i] = new_position.copy()

                if func(new_position) < func(gbest):
                    gbest = new_position.copy()

            simplex = [gbest + np.random.normal(0, 0.5, self.dim) for _ in range(self.dim + 1)]
            gbest = self.optimize_simplex(simplex, func)

            if it % self.w_change_iter == 0 and it > 0:
                inertial_weight = self.inertia_max - (self.inertia_max - self.inertia_min) * it / self.max_iter

        return gbest
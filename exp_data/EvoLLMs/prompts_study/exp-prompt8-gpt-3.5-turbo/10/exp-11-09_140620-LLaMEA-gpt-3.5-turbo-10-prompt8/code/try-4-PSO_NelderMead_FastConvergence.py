class PSO_NelderMead_FastConvergence(PSO_NelderMead):
    def __init__(self, budget, dim, swarm_size=30, max_iter=100, inertia=0.5, c1=2.0, c2=2.0):
        super().__init__(budget, dim, swarm_size, max_iter)
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2

    def __call__(self, func):
        # Initialize algorithm parameters and variables
        inertia_min, inertia_max = 0.3, 0.9
        swarm = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        pbest = swarm.copy()
        velocity = np.zeros((self.swarm_size, self.dim))
        gbest_idx = np.argmin([func(p) for p in swarm])
        gbest = swarm[gbest_idx].copy()
        inertia = self.inertia

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

            # Dynamic inertia weight adaptation based on fitness improvement
            prev_gbest_fitness = func(gbest)
            prev_inertia = inertia

            if func(gbest) < prev_gbest_fitness:
                inertia = min(inertia_max, inertia + 0.1)
            else:
                inertia = max(inertia_min, inertia - 0.1)

            velocity *= prev_inertia / inertia

        return gbest
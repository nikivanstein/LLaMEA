class Enhanced_PSO_NelderMead:
    def __init__(self, budget, dim, swarm_size=30, max_iter=100, inertia_max=0.9, inertia_min=0.4):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.lb = -5.0
        self.ub = 5.0
        self.inertia_max = inertia_max
        self.inertia_min = inertia_min

    def __call__(self, func):
        inertia_weight = self.inertia_max
        # Rest of the code remains the same

        for _ in range(self.max_iter):
            for i in range(self.swarm_size):
                # Existing code for velocity update and position update

                if func(new_position) < func(pbest[i]):
                    pbest[i] = new_position.copy()

                swarm[i] = new_position.copy()

                if func(new_position) < func(gbest):
                    gbest = new_position.copy()

            # Introduce dynamic inertia weight adaptation
            inertia_weight = self.inertia_max - ((_ + 1) / self.max_iter) * (self.inertia_max - self.inertia_min)

            simplex = [gbest + np.random.normal(0, 0.5, self.dim) for _ in range(self.dim + 1)]
            gbest = self.optimize_simplex(simplex, func, inertia_weight)

        return gbest
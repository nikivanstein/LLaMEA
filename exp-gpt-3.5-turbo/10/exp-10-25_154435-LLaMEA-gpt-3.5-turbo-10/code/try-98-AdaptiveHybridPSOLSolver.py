class AdaptiveHybridPSOLSolver:
    def __init__(self, budget, dim, swarm_size=30, omega=0.5, alpha=0.1):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.omega = omega
        self.alpha = alpha

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def local_search(x, f_x, mutation_rate):
            for _ in range(5):
                x_new = x + np.random.uniform(-mutation_rate, mutation_rate, size=self.dim)
                f_x_new = objective_function(x_new)
                if f_x_new < f_x:
                    x, f_x = x_new, f_x_new
            return x, f_x

        swarm = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        swarm_cost = np.array([objective_function(x) for x in swarm])
        g_best_idx = np.argmin(swarm_cost)
        g_best = swarm[g_best_idx]
        
        mutation_rate = self.alpha

        for _ in range(self.budget - self.swarm_size):
            for i in range(self.swarm_size):
                p_best = swarm[i] if swarm_cost[i] < objective_function(swarm[i]) else g_best
                r_p = np.random.rand(self.dim)
                swarm[i] = self.omega * swarm[i] + r_p * (p_best - swarm[i])
                swarm[i] = np.clip(swarm[i], -5.0, 5.0)
                swarm[i], swarm_cost[i] = local_search(swarm[i], swarm_cost[i], mutation_rate)
                if swarm_cost[i] < objective_function(g_best):
                    g_best = swarm[i]

            # Update mutation rate adaptively based on the best cost
            mutation_rate *= 0.95 if swarm_cost[i] < objective_function(g_best) else 1.05

        return g_best
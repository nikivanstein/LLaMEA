import numpy as np

class AdaptivePSOLSolver:
    def __init__(self, budget, dim, swarm_size=30, omega=0.5, phi_p=0.5, phi_g=0.5, alpha=0.1, mutation_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.omega = omega
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.alpha = alpha
        self.mutation_prob = mutation_prob

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def local_search(x, f_x):
            for _ in range(5):
                x_new = x + np.random.uniform(-self.alpha, self.alpha, size=self.dim)
                f_x_new = objective_function(x_new)
                if f_x_new < f_x:
                    x, f_x = x_new, f_x_new
            return x, f_x

        swarm = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        swarm_cost = np.array([objective_function(x) for x in swarm])
        g_best_idx = np.argmin(swarm_cost)
        g_best = swarm[g_best_idx]
        
        for _ in range(self.budget - self.swarm_size):
            for i in range(self.swarm_size):
                p_best = swarm[i] if swarm_cost[i] < objective_function(swarm[i]) else g_best
                r_p, r_g = np.random.rand(self.dim), np.random.rand(self.dim)
                swarm[i] = self.omega * swarm[i] + self.phi_p * r_p * (p_best - swarm[i]) + self.phi_g * r_g * (g_best - swarm[i])
                swarm[i] = np.clip(swarm[i], -5.0, 5.0)
                
                if np.random.rand() < self.mutation_prob:
                    swarm[i] += np.random.uniform(-self.alpha, self.alpha, size=self.dim)
                    swarm[i] = np.clip(swarm[i], -5.0, 5.0)
                    
                swarm[i], swarm_cost[i] = local_search(swarm[i], swarm_cost[i])
                if swarm_cost[i] < objective_function(g_best):
                    g_best = swarm[i]

        return g_best
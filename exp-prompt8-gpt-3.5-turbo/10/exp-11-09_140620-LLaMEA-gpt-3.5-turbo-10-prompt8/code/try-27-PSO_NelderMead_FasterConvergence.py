import numpy as np

class PSO_NelderMead_FasterConvergence:
    def __init__(self, budget, dim, swarm_size=30, max_iter=100, inertia_weight=0.5, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.lb = -5.0
        self.ub = 5.0
        self.inertia_weight = inertia_weight
        self.c1 = c1
        self.c2 = c2

    def optimize_simplex(self, simplex, func):
        # Same as before

    def __call__(self, func):
        swarm = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        pbest = swarm.copy()
        velocity = np.zeros((self.swarm_size, self.dim))
        gbest_idx = np.argmin([func(p) for p in swarm])
        gbest = swarm[gbest_idx].copy()
        prev_best_fitness = func(gbest)

        for _ in range(self.max_iter):
            for i in range(self.swarm_size):
                new_velocity = self.inertia_weight * velocity[i] + self.c1 * np.random.rand() * (pbest[i] - swarm[i]) + self.c2 * np.random.rand() * (gbest - swarm[i])
                new_position = swarm[i] + new_velocity
                new_position = np.clip(new_position, self.lb, self.ub)

                if func(new_position) < func(pbest[i]):
                    pbest[i] = new_position.copy()

                swarm[i] = new_position.copy()

                current_best_fitness = func(gbest)
                if func(new_position) < current_best_fitness:
                    gbest = new_position.copy()
                    fitness_gain = prev_best_fitness - current_best_fitness
                    self.inertia_weight = max(0.4, min(self.inertia_weight + 0.1 * fitness_gain, 0.9))
                    prev_best_fitness = current_best_fitness

            simplex = [gbest + np.random.normal(0, 0.5, self.dim) for _ in range(self.dim + 1)]
            gbest = self.optimize_simplex(simplex, func)

        return gbest
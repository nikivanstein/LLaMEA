import numpy as np

class Enhanced_PSO_NelderMead:
    def __init__(self, budget, dim, swarm_size=30, max_iter=100):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.lb = -5.0
        self.ub = 5.0

    def optimize_simplex(self, simplex, func):
        for _ in range(self.budget // self.dim):
            simplex.sort(key=lambda x: func(x))
            centroid = np.mean(simplex[:-1], axis=0)
            reflection = centroid + (centroid - simplex[-1])
            if func(simplex[0]) <= func(reflection) < func(simplex[-2]):
                simplex[-1] = reflection
            elif func(reflection) < func(simplex[0]):
                expansion = centroid + 2*(reflection - centroid)
                if func(expansion) < func(reflection):
                    simplex[-1] = expansion
                else:
                    simplex[-1] = reflection
            else:
                contraction = centroid + 0.5*(simplex[-1] - centroid)
                if func(contraction) < func(simplex[-1]):
                    simplex[-1] = contraction
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = 0.5*(simplex[i] + simplex[0])

        return simplex[0]

    def __call__(self, func):
        swarm = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        pbest = swarm.copy()
        velocity = np.zeros((self.swarm_size, self.dim))
        gbest_idx = np.argmin([func(p) for p in swarm])
        gbest = swarm[gbest_idx].copy()
        inertia_weight = 0.5

        for _ in range(self.max_iter):
            for i in range(self.swarm_size):
                new_velocity = inertia_weight * velocity[i] + np.random.rand() * (pbest[i] - swarm[i]) + np.random.rand() * (gbest - swarm[i])
                new_position = swarm[i] + new_velocity
                new_position = np.clip(new_position, self.lb, self.ub)

                if func(new_position) < func(pbest[i]):
                    pbest[i] = new_position.copy()

                swarm[i] = new_position.copy()

                if func(new_position) < func(gbest):
                    gbest = new_position.copy()

            simplex = [gbest + np.random.normal(0, 0.5, self.dim) for _ in range(self.dim + 1)]
            gbest_prev = gbest.copy()
            gbest = self.optimize_simplex(simplex, func)

            if func(gbest) < func(gbest_prev):
                inertia_weight = max(0.4, inertia_weight * 1.1)  # Dynamic inertia weight adjustment

        return gbest
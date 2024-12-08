import numpy as np

class DynamicLocalSearchPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.a = 1.5
        self.search_radius = 1.0

    def dynamic_local_search(self, func, point):
        fitness = func(point)
        for _ in range(10):
            candidate_point = point + np.random.uniform(-self.search_radius, self.search_radius, self.dim)
            candidate_fitness = func(candidate_point)
            if candidate_fitness < fitness:
                point = candidate_point
                fitness = candidate_fitness
        return point, fitness

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.dim, self.dim))
        velocity = np.zeros((self.dim, self.dim))
        pbest = swarm.copy()
        pbest_fitness = np.apply_along_axis(func, 1, pbest)
        gbest = pbest[np.argmin(pbest_fitness)]
        gbest_fitness = np.min(pbest_fitness)

        for _ in range(self.budget):
            r1 = np.random.random((self.dim, self.dim))
            r2 = np.random.random((self.dim, self.dim))

            velocity = self.w * velocity + self.c1 * r1 * (pbest - swarm) + self.c2 * r2 * (gbest - swarm)
            swarm += velocity

            updated_swarm = np.array([self.dynamic_local_search(func, point) for point in swarm])
            swarm, fitness = updated_swarm[:, 0], updated_swarm[:, 1]

            update_indices = fitness < pbest_fitness
            pbest[update_indices] = swarm[update_indices]
            pbest_fitness[update_indices] = fitness[update_indices]

            if np.min(fitness) < gbest_fitness:
                gbest = swarm[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)

            self.w = 0.4 + 0.5 * (1 - _ / self.budget)
            self.c1 = max(1.5 - 0.5 * _ / self.budget, 1.0)
            self.c2 = min(1.5 + 0.5 * _ / self.budget, 3.0)
            self.search_radius = 1.0 / (1 + 0.1 * _)

        return gbest_fitness
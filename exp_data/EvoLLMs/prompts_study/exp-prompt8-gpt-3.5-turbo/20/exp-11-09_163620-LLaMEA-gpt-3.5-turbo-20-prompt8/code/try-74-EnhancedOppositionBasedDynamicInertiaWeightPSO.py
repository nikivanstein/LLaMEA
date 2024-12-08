class EnhancedOppositionBasedDynamicInertiaWeightPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.base_mutation_rate = 0.1
        self.mutation_rate = self.base_mutation_rate
        self.inertia_weights = np.full(dim, self.w_max)

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.dim, self.dim))
        velocity = np.zeros((self.dim, self.dim))
        pbest = swarm.copy()
        pbest_fitness = np.apply_along_axis(func, 1, pbest)
        gbest = pbest[np.argmin(pbest_fitness)]
        gbest_fitness = np.min(pbest_fitness)

        for t in range(1, self.budget + 1):
            r1 = np.random.random((self.dim, self.dim))
            r2 = np.random.random((self.dim, self.dim))

            velocity = self.inertia_weights[:, np.newaxis] * velocity + self.c1 * r1 * (pbest - swarm) + self.c2 * r2 * (gbest - swarm)
            swarm += velocity

            fitness = np.apply_along_axis(func, 1, swarm)
            update_indices = fitness < pbest_fitness
            pbest[update_indices] = swarm[update_indices]
            pbest_fitness[update_indices] = fitness[update_indices]

            opposite_swarm = 2 * gbest - swarm
            opposite_fitness = np.apply_along_axis(func, 1, opposite_swarm)
            update_indices = opposite_fitness < pbest_fitness
            pbest[update_indices] = opposite_swarm[update_indices]
            pbest_fitness[update_indices] = opposite_fitness[update_indices]

            if t % (self.budget // 5) == 0:
                improvement_rate = (gbest_fitness - np.min(fitness)) / gbest_fitness
                self.inertia_weights = np.clip(self.inertia_weights * (1.0 + 0.1 * improvement_rate), self.w_min, self.w_max)

            if np.min(fitness) < gbest_fitness:
                gbest = swarm[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)

        return gbest_fitness
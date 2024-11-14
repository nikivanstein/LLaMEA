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

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.dim, self.dim))
        velocity = np.zeros((self.dim, self.dim))
        pbest = swarm.copy()
        pbest_fitness = np.apply_along_axis(func, 1, pbest)
        gbest = pbest[np.argmin(pbest_fitness)]
        gbest_fitness = np.min(pbest_fitness)

        w = self.w_max

        for t in range(1, self.budget + 1):
            r1 = np.random.random((self.dim, self.dim))
            r2 = np.random.random((self.dim, self.dim))

            velocity = w * velocity + self.c1 * r1 * (pbest - swarm) + self.c2 * r2 * (gbest - swarm)
            swarm += velocity

            fitness = np.apply_along_axis(func, 1, swarm)
            update_indices = fitness < pbest_fitness
            pbest[update_indices] = swarm[update_indices]
            pbest_fitness[update_indices] = fitness[update_indices]

            # Opposition-based Learning on personal best positions
            opposite_pbest = 2 * pbest - swarm
            opposite_pbest_fitness = np.apply_along_axis(func, 1, opposite_pbest)
            update_indices = opposite_pbest_fitness < pbest_fitness
            pbest[update_indices] = opposite_pbest[update_indices]
            pbest_fitness[update_indices] = opposite_pbest_fitness[update_indices]

            # Opposition-based Learning on global best position
            opposite_gbest = 2 * gbest - swarm
            opposite_gbest_fitness = np.apply_along_axis(func, 1, opposite_gbest)
            update_indices = opposite_gbest_fitness < pbest_fitness
            pbest[update_indices] = opposite_gbest[update_indices]
            pbest_fitness[update_indices] = opposite_gbest_fitness[update_indices]

            # Dynamic Mutation
            if t % (self.budget // 5) == 0:  # Adjust mutation rate every 20% of the budget
                improvement_rate = (gbest_fitness - np.min(fitness)) / gbest_fitness
                self.mutation_rate = self.base_mutation_rate + 0.5 * improvement_rate

            mutation_indices = np.random.choice(self.dim, int(self.dim * self.mutation_rate), replace=False)
            swarm[mutation_indices] = np.random.uniform(-5.0, 5.0, (len(mutation_indices), self.dim))

            if np.min(fitness) < gbest_fitness:
                gbest = swarm[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)

            w = self.w_min + (t / self.budget) * (self.w_max - self.w_min)

        return gbest_fitness
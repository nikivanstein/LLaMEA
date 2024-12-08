import numpy as np

class AdaptiveOppositionBasedDynamicInertiaWeightPSO:
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

            # Opposition-based Learning
            opposite_swarm = 2 * gbest - swarm
            opposite_fitness = np.apply_along_axis(func, 1, opposite_swarm)
            update_indices = opposite_fitness < pbest_fitness
            pbest[update_indices] = opposite_swarm[update_indices]
            pbest_fitness[update_indices] = opposite_fitness[update_indices]

            # Adaptive Local Search Operator
            if t % (self.budget // 10) == 0:  # Introduce local search every 10% of the budget
                local_search_indices = np.argsort(fitness)[:self.dim // 10]
                local_search_swarm = swarm[local_search_indices]
                local_search_swarm += np.random.normal(0, 0.1, local_search_swarm.shape)
                local_search_fitness = np.apply_along_axis(func, 1, local_search_swarm)

                update_indices = local_search_fitness < fitness[local_search_indices]
                swarm[local_search_indices[update_indices]] = local_search_swarm[update_indices]

            if np.min(fitness) < gbest_fitness:
                gbest = swarm[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)

            w = self.w_min + (t / self.budget) * (self.w_max - self.w_min)

        return gbest_fitness
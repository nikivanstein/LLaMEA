import numpy as np

class AdaptiveMutationDynamicInertiaWeightPSO:
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

            # Novel Adaptive Mutation Strategy
            if t % (self.budget // 5) == 0:  # Adjust mutation rate every 20% of the budget
                improvement_rate = (gbest_fitness - np.min(fitness)) / gbest_fitness
                self.mutation_rate = self.base_mutation_rate + 0.5 * improvement_rate

            mutation_indices = np.random.choice(self.dim, int(self.dim * self.mutation_rate), replace=False)
            mutation_swarm = np.copy(swarm)
            for i in range(len(mutation_indices)):
                candidate = np.copy(swarm[i])
                chosen_indices = np.random.choice(self.dim, 2, replace=False)
                trial_vector = swarm[chosen_indices[0]] + np.random.uniform(-1, 1) * (swarm[chosen_indices[1]] - swarm[i])
                for j in range(self.dim):
                    if np.random.uniform() < self.mutation_rate:
                        candidate[j] = trial_vector[j]
                mutation_swarm[i] = candidate
            
            mutation_fitness = np.apply_along_axis(func, 1, mutation_swarm)
            update_indices = mutation_fitness < pbest_fitness
            pbest[update_indices] = mutation_swarm[update_indices]
            pbest_fitness[update_indices] = mutation_fitness[update_indices]

            if np.min(mutation_fitness) < gbest_fitness:
                gbest = mutation_swarm[np.argmin(mutation_fitness)]
                gbest_fitness = np.min(mutation_fitness)

            w = self.w_min + (t / self.budget) * (self.w_max - self.w_min)

        return gbest_fitness
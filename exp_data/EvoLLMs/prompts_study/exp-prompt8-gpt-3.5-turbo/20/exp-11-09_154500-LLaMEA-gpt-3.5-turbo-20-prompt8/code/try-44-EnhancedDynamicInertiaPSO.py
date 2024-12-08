import numpy as np

class EnhancedDynamicInertiaPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.pop_size = 20
        self.max_velocity = 0.1 * (5.0 - (-5.0))
        self.mutation_prob = 0.1
        self.chaos_param = 0.8

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        velocity = np.zeros((self.pop_size, self.dim))
        pbest = population.copy()
        pbest_fitness = np.array([func(ind) for ind in population])
        gbest_idx = np.argmin(pbest_fitness)
        gbest = pbest[gbest_idx]

        for _ in range(self.budget):
            w = self.w_max - (_ / self.budget) * (self.w_max - self.w_min)
            r1, r2 = np.random.rand(), np.random.rand()

            chaos_map = lambda x: self.chaos_param * x * (1 - x)
            chaos_func = np.vectorize(chaos_map)
            mutation_mask = np.where(np.random.rand(self.pop_size, self.dim) < self.mutation_prob, 1, 0)
            mutation_values = chaos_func(np.random.rand(self.pop_size, self.dim)) * mutation_mask

            velocity = w * velocity + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (np.tile(gbest, (self.pop_size, 1)) - population) + mutation_values
            velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
            population += velocity

            fitness = np.array([func(ind) for ind in population])
            update_indices = fitness < pbest_fitness
            pbest[update_indices] = population[update_indices]
            pbest_fitness[update_indices] = fitness[update_indices]

            gbest_idx = np.argmin(pbest_fitness)
            gbest = pbest[gbest_idx]

        return gbest
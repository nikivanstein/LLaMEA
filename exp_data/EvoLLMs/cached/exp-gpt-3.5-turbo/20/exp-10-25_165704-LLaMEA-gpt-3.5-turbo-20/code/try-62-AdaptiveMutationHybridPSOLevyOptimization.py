import numpy as np

class AdaptiveMutationHybridPSOLevyOptimization(HybridPSOLevyOptimization):
    def __init__(self, budget, dim, pop_size=30, c1=1.496, c2=1.496, w=0.729, alpha=1.5, beta=1.5, mutation_prob=0.2):
        super().__init__(budget, dim, pop_size, c1, c2, w, alpha, beta)
        self.mutation_prob = mutation_prob

    def __call__(self, func):
        def levy_flight():
            return np.random.standard_cauchy(self.dim) / (np.random.gamma(self.alpha, 1/self.beta, self.dim) ** (1/self.beta))

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        pbest = population.copy()
        pbest_vals = np.array([func(ind) for ind in pbest])
        gbest = pbest[np.argmin(pbest_vals)]
        gbest_val = np.min(pbest_vals)

        for _ in range(self.budget):
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            velocities = self.w * velocities + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (gbest - population)
            population += velocities

            for i in range(self.pop_size):
                if np.any(population[i] < -5.0) or np.any(population[i] > 5.0):
                    population[i] = np.clip(population[i], -5.0, 5.0)

            for i in range(self.pop_size):
                if func(population[i]) < pbest_vals[i]:
                    pbest[i] = population[i]
                    pbest_vals[i] = func(population[i])

            new_gbest_val = np.min(pbest_vals)
            if new_gbest_val < gbest_val:
                gbest = pbest[np.argmin(pbest_vals)]
                gbest_val = new_gbest_val

            # Levy flight exploration
            for i in range(self.pop_size):
                if np.random.rand() < self.mutation_prob:
                    population[i] += 0.01 * levy_flight()

        return gbest
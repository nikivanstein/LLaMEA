import numpy as np

class AdaptiveParticleSwarmOptimization:
    def __init__(self, budget, dim, pop_size=30, w=0.5, c1=2.0, c2=2.0):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        pbest = population.copy()
        pbest_vals = np.array([func(ind) for ind in pbest])
        gbest = pbest[np.argmin(pbest_vals)]
        gbest_val = np.min(pbest_vals)

        for _ in range(self.budget):
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = self.w * velocities[i] + self.c1 * r1 * (pbest[i] - population[i]) + self.c2 * r2 * (gbest - population[i])
                population[i] = np.clip(population[i] + velocities[i], -5.0, 5.0)

                if func(population[i]) < pbest_vals[i]:
                    pbest[i] = population[i]
                    pbest_vals[i] = func(population[i])

            new_gbest_val = np.min(pbest_vals)
            if new_gbest_val < gbest_val:
                gbest = pbest[np.argmin(pbest_vals)]
                gbest_val = new_gbest_val

            # Adaptive inertia weight
            self.w = 1 / (1 + np.exp(-np.random.normal(0.5, 0.1)))

            # Dynamic cognitive and social parameters
            self.c1 = 1.5 if np.random.rand() < 0.2 else 2.0
            self.c2 = 1.5 if np.random.rand() < 0.2 else 2.0

        return gbest
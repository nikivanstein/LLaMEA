import numpy as np
from scipy.stats import cauchy

class DynamicPopulationPSO:
    def __init__(self, budget, dim, swarm_size=30, max_iter=100):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.max_iter = max_iter

    def __call__(self, func):
        def chaotic_init():
            return np.random.uniform(-5.0, 5.0, self.dim)

        def objective(x):
            return func(x)

        swarm = np.array([chaotic_init() for _ in range(self.swarm_size)])
        velocity = np.zeros((self.swarm_size, self.dim))
        pbest = swarm.copy()
        pbest_fitness = np.array([objective(ind) for ind in pbest])
        gbest_idx = np.argmin(pbest_fitness)
        gbest = pbest[gbest_idx].copy()
        gbest_fitness = pbest_fitness[gbest_idx]
        
        inertia_weight = 0.9  # Initialize inertia weight
        swarm_reduction = int(0.1 * self.max_iter)  # Number of iterations after which we reduce swarm

        for t in range(self.max_iter):
            if t % swarm_reduction == 0 and self.swarm_size > 5:
                self.swarm_size -= 5
                swarm = swarm[:self.swarm_size]
                velocity = velocity[:self.swarm_size]
                pbest = pbest[:self.swarm_size]
                pbest_fitness = pbest_fitness[:self.swarm_size]

            inertia_weight -= 0.8 / self.max_iter
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                if np.random.rand() < 0.1:  # 10% chance for Levy flight
                    step = 0.01 * cauchy.rvs(size=self.dim)  # Levy flight step
                    velocity[i] = step
                    swarm[i] += velocity[i]
                else:
                    velocity[i] = inertia_weight * velocity[i] + 2.0 * r1 * (pbest[i] - swarm[i]) + 2.0 * r2 * (gbest - swarm[i])
                    swarm[i] += velocity[i]
                swarm[i] = np.clip(swarm[i], -5.0, 5.0)
                fitness = objective(swarm[i])
                if fitness < pbest_fitness[i]:
                    pbest[i] = swarm[i].copy()
                    pbest_fitness[i] = fitness
                    if fitness < gbest_fitness:
                        gbest = swarm[i].copy()
                        gbest_fitness = fitness

        return gbest
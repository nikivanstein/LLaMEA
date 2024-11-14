import numpy as np
from scipy.stats import cauchy

class EnhancedPSO:
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

        for _ in range(self.max_iter):
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
                    # Opposition-based learning
                    opp_swarm = 2 * np.mean(swarm) - swarm
                    opp_fitness = objective(opp_swarm)
                    if opp_fitness < pbest_fitness[i]:
                        pbest[i] = opp_swarm.copy()
                        pbest_fitness[i] = opp_fitness
                        if opp_fitness < gbest_fitness:
                            gbest = opp_swarm.copy()
                            gbest_fitness = opp_fitness
                swarm[i] = np.clip(swarm[i], -5.0, 5.0)

        return gbest
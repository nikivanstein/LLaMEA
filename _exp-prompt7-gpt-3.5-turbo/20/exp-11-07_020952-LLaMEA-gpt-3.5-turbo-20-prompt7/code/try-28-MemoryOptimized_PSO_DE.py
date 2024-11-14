import numpy as np

class MemoryOptimized_PSO_DE:
    def __init__(self, budget, dim, swarm_size=30, pso_weight=0.5, c1=1.5, c2=2.0, de_weight=0.8, de_cr=0.7):
        self.budget, self.dim, self.swarm_size = budget, dim, swarm_size
        self.pso_weight, self.c1, self.c2, self.de_weight, self.de_cr = pso_weight, c1, c2, de_weight, de_cr
        self.rand_pso = self.pso_weight * np.random.uniform(0, 1, size=(self.swarm_size, self.dim))
        self.rand_de = self.de_weight * np.random.uniform(0, 1, size=(self.swarm_size, self.dim))

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        velocity = np.zeros((self.swarm_size, self.dim))
        pbest = swarm.copy()
        gbest = min(pbest, key=func)

        for _ in range(self.budget):
            for i in range(self.swarm_size):
                x, v, p = swarm[i], velocity[i], pbest[i]
                r1, r2 = np.random.uniform(0, 1, size=(2, self.dim))
                p_r1_c1 = self.rand_pso[i] + self.c1 * r1
                p_r2_c2_gbest = self.c2 * r2 * (gbest - x)
                swarm[i], velocity[i] = np.clip(x + p_r1_c1 + p_r2_c2_gbest, -5.0, 5.0), v
                r1, r2, r3 = np.random.choice(self.swarm_size, 3, replace=False)
                pbest_r1_r2_r3 = pbest[r1] + self.rand_de[r1] * (pbest[r2] - pbest[r3])
                trial = np.clip(pbest_r1_r2_r3, -5.0, 5.0)
                pbest[i] = trial if func(trial) < func(p) else p

            gbest = min(pbest, key=func)

        return gbest
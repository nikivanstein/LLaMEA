import numpy as np

class Enhanced_PSO_DE_Optimized:
    def __init__(self, budget, dim, swarm_size=30, pso_weight=0.5, c1=1.5, c2=2.0, de_weight=0.8, de_cr=0.7):
        self.budget, self.dim, self.swarm_size = budget, dim, swarm_size
        self.pso_weight, self.c1, self.c2, self.de_weight, self.de_cr = pso_weight, c1, c2, de_weight, de_cr
        self.rand_pso = self.pso_weight * np.random.uniform(0, 1, size=(self.swarm_size, self.dim))
        self.rand_de = self.de_weight * np.random.uniform(0, 1, size=(self.swarm_size, self.dim))
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        def limit_bounds(x):
            np.clip(x, self.bounds[0], self.bounds[1], out=x)

        swarm = np.random.uniform(*self.bounds, size=(self.swarm_size, self.dim))
        velocity = np.zeros((self.swarm_size, self.dim))
        pbest = np.copy(swarm)
        gbest = min(pbest, key=func)

        for _ in range(self.budget):
            for i in range(self.swarm_size):
                x, v, p = swarm[i], velocity[i], pbest[i]
                r1, r2 = np.random.uniform(0, 1, size=(2, self.dim))
                rand_pso_i, rand_de_i = self.rand_pso[i], self.rand_de[i]
                limit_bounds(x + rand_pso_i + self.c1 * r1 * (p - x) + self.c2 * r2 * (gbest - x))
                v.fill(0)
                r1, r2, r3 = np.random.choice(self.swarm_size, 3, replace=False)
                pbest_r1, pbest_r2, pbest_r3 = pbest[r1], pbest[r2], pbest[r3]
                trial = pbest_r1 + rand_de_i * (pbest_r2 - pbest_r3)
                if func(trial) < func(p):
                    np.copyto(p, trial)

            gbest = min(pbest, key=func)

        return gbest
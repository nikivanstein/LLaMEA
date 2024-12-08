import numpy as np

class Enhanced_PSO_DE_Optimized:
    def __init__(self, budget, dim, swarm_size=30, pso_weight=0.5, c1=1.5, c2=2.0, de_weight=0.8, de_cr=0.7):
        self.budget, self.dim, self.swarm_size = budget, dim, swarm_size
        self.pso_weight, self.c1, self.c2, self.de_weight, self.de_cr = pso_weight, c1, c2, de_weight, de_cr
        self.rand_pso = self.pso_weight * np.random.uniform(0, 1, size=(self.swarm_size, self.dim))
        self.rand_de = self.de_weight * np.random.uniform(0, 1, size=(self.swarm_size, self.dim))
        self.r1_r2 = np.random.uniform(0, 1, size=(2, self.dim))
        self.rand_choice = np.random.choice(self.swarm_size, 3, replace=False)

    def __call__(self, func):
        def limit_bounds(x):
            np.clip(x, -5.0, 5.0, out=x)

        swarm = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        velocity = np.zeros((self.swarm_size, self.dim))
        pbest = np.copy(swarm)
        gbest = min(pbest, key=func)

        for _ in range(self.budget):
            for i in range(self.swarm_size):
                x, v, p = swarm[i], velocity[i], pbest[i]
                rand_pso_i, rand_de_i = self.rand_pso[i], self.rand_de[i]
                limit_bounds(x + rand_pso_i + self.c1 * self.r1_r2[0] * (p - x) + self.c2 * self.r1_r2[1] * (gbest - x))
                np.copyto(v, 0)
                pbest_r = pbest[self.rand_choice]
                trial = np.copy(pbest_r[0] + rand_de_i * (pbest_r[1] - pbest_r[2]))
                if func(trial) < func(p):
                    np.copyto(p, trial)

            gbest = min(pbest, key=func)

        return gbest
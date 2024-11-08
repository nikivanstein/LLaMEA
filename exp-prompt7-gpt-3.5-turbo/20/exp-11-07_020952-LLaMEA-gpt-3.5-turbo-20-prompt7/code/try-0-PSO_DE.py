import numpy as np

class PSO_DE:
    def __init__(self, budget, dim, swarm_size=30, pso_weight=0.5, c1=1.5, c2=2.0, de_weight=0.8, de_cr=0.7):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.pso_weight = pso_weight
        self.c1 = c1
        self.c2 = c2
        self.de_weight = de_weight
        self.de_cr = de_cr

    def __call__(self, func):
        def limit_bounds(x):
            return np.clip(x, -5.0, 5.0)

        def pso_update(x, v, pbest, gbest):
            r1 = np.random.uniform(0, 1, size=(self.swarm_size, self.dim))
            r2 = np.random.uniform(0, 1, size=(self.swarm_size, self.dim))

            v = self.pso_weight * v + self.c1 * r1 * (pbest - x) + self.c2 * r2 * (gbest - x)
            x = limit_bounds(x + v)

            return x, v

        def de_update(x, pbest):
            r1, r2, r3 = np.random.choice(self.swarm_size, 3, replace=False)

            mutant = limit_bounds(pbest[r1] + self.de_weight * (pbest[r2] - pbest[r3]))
            trial = np.where(np.random.uniform(0, 1, self.dim) < self.de_cr, mutant, x)

            return trial

        swarm = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        velocity = np.zeros((self.swarm_size, self.dim))
        pbest = swarm.copy()
        gbest = pbest[np.argmin([func(p) for p in pbest])]

        for _ in range(self.budget):
            for i in range(self.swarm_size):
                x = swarm[i]
                v = velocity[i]

                x, v = pso_update(x, v, pbest[i], gbest)
                
                trial = de_update(x, pbest)
                if func(trial) < func(pbest[i]):
                    pbest[i] = trial

            gbest = pbest[np.argmin([func(p) for p in pbest])]

        return gbest
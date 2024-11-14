import numpy as np

class Enhanced_PSO_DE_Optimized:
    def __init__(self, budget, dim, swarm_size=30, pso_weight=0.5, c1=1.5, c2=2.0, de_weight=0.8, de_cr=0.7):
        self.budget, self.dim, self.swarm_size = budget, dim, swarm_size
        self.pso_weight, self.c1, self.c2, self.de_weight, self.de_cr = pso_weight, c1, c2, de_weight, de_cr
        self.rand_pso = self.pso_weight * np.random.uniform(0, 1, size=(self.swarm_size, self.dim))
        self.rand_de = self.de_weight * np.random.uniform(0, 1, size=(self.swarm_size, self.dim))

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        velocity = np.zeros((self.swarm_size, self.dim))
        pbest = np.copy(swarm)
        gbest = pbest[np.argmin([func(p) for p in pbest])]

        for _ in range(self.budget):
            r1, r2 = np.random.uniform(0, 1, size=(2, self.swarm_size, self.dim))
            rand_pso = self.rand_pso[:, np.newaxis, :]
            rand_de = self.rand_de[:, np.newaxis, :]
            c1r1 = self.c1 * r1 * (pbest - swarm)
            c2r2 = self.c2 * r2 * (gbest - swarm)
            swarm += rand_pso + c1r1 + c2r2
            np.clip(swarm, -5.0, 5.0, out=swarm)

            pbest_idx = np.argmin([func(p) for p in pbest])
            r1, r2, r3 = np.random.choice(self.swarm_size, (3, self.dim))
            pbest_r1, pbest_r2, pbest_r3 = pbest[[r1, r2, r3]]
            trial = pbest_r1 + self.rand_de * (pbest_r2 - pbest_r3)
            better_mask = func(trial) < [func(p) for p in pbest]
            pbest[better_mask] = trial[better_mask]

            gbest = pbest[np.argmin([func(p) for p in pbest])]

        return gbest
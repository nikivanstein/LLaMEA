import numpy as np

class PSO_DE:
    def __init__(self, budget, dim, swarm_size=30, pso_weight=0.5, c1=1.5, c2=2.0, de_weight=0.8, de_cr=0.7):
        self.budget, self.dim, self.swarm_size, self.pso_weight, self.c1, self.c2, self.de_weight, self.de_cr = budget, dim, swarm_size, pso_weight, c1, c2, de_weight, de_cr

    def __call__(self, func):
        def limit_bounds(x):
            return np.clip(x, -5.0, 5.0)

        def update_pso_de(x, v, pbest, gbest, r1, r2):
            v = self.pso_weight * v + self.c1 * r1 * (pbest - x) + self.c2 * r2 * (gbest - x)
            x = limit_bounds(x + v)
            return x, v

        swarm = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        velocity = np.zeros((self.swarm_size, self.dim))
        pbest = swarm.copy()
        gbest = pbest[np.argmin([func(p) for p in pbest])

        for _ in range(self.budget):
            for i in range(self.swarm_size):
                x, v = swarm[i], velocity[i]

                r1, r2 = np.random.uniform(0, 1, size=(2, self.dim))
                x, v = update_pso_de(x, v, pbest[i], gbest, r1, r2)

                r1, r2, r3 = np.random.choice(self.swarm_size, 3, replace=False)
                mutant = limit_bounds(pbest[r1] + self.de_weight * (pbest[r2] - pbest[r3]))
                trial = np.where(np.random.uniform(0, 1, self.dim) < self.de_cr, mutant, x)

                if func(trial) < func(pbest[i]):
                    pbest[i] = trial

            gbest = pbest[np.argmin([func(p) for p in pbest])]

        return gbest
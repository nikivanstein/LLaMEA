import numpy as np

class Enhanced_PSO_DE:
    def __init__(self, budget, dim, swarm_size=30, pso_weight=0.5, c1=1.5, c2=2.0, de_weight=0.8, de_cr=0.7):
        self.budget, self.dim, self.swarm_size, self.pso_weight, self.c1, self.c2, self.de_weight, self.de_cr = budget, dim, swarm_size, pso_weight, c1, c2, de_weight, de_cr

    def __call__(self, func):
        def limit_bounds(x):
            return np.clip(x, -5.0, 5.0)

        swarm = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        velocity = np.zeros((self.swarm_size, self.dim))
        pbest = swarm.copy()
        gbest = min(pbest, key=func)

        for _ in range(self.budget):
            for i in range(self.swarm_size):
                x = swarm[i]
                v = velocity[i]
                p = pbest[i]
                r1, r2 = np.random.uniform(0, 1, size=(2, self.dim))
                swarm[i], velocity[i] = limit_bounds(x + self.pso_weight * v + self.c1 * r1 * (p - x) + self.c2 * r2 * (gbest - x)), v
                r1, r2, r3 = np.random.choice(self.swarm_size, 3, replace=False)
                trial = limit_bounds(pbest[r1] + self.de_weight * (pbest[r2] - pbest[r3]))
                pbest[i] = trial if func(trial) < func(p) else p

            gbest = min(pbest, key=func)

        return gbest
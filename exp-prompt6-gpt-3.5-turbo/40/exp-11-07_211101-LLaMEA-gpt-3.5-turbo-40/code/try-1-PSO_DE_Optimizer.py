import numpy as np

class PSO_DE_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, de_cr=0.5, de_f=0.8, w=0.5, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.de_cr = de_cr
        self.de_f = de_f
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def __call__(self, func):
        def de_mutate(pop, target_idx):
            candidates = [idx for idx in range(len(pop)) if idx != target_idx]
            a, b, c = pop[np.random.choice(candidates, 3, replace=False)]
            return np.clip(a + self.de_f * (b - c), -5.0, 5.0)

        def pso_de_optimize():
            swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
            velocity = np.zeros((self.swarm_size, self.dim))
            pbest = swarm.copy()
            pbest_fitness = np.array([func(ind) for ind in swarm])
            gbest_idx = np.argmin(pbest_fitness)
            gbest = swarm[gbest_idx]

            for _ in range(self.budget):
                for i in range(self.swarm_size):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    velocity[i] = self.w * velocity[i] + self.c1 * r1 * (pbest[i] - swarm[i]) + self.c2 * r2 * (gbest - swarm[i])
                    swarm[i] = np.clip(swarm[i] + velocity[i], -5.0, 5.0)

                    trial_vector = de_mutate(swarm, i)
                    trial_fitness = func(trial_vector)
                    if trial_fitness < pbest_fitness[i]:
                        pbest[i] = trial_vector
                        pbest_fitness[i] = trial_fitness
                        if trial_fitness < pbest_fitness[gbest_idx]:
                            gbest_idx = i
                            gbest = trial_vector

            return gbest

        return pso_de_optimize()
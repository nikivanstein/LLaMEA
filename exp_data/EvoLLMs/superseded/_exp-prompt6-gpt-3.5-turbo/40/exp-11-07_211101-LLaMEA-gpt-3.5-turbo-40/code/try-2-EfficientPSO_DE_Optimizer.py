import numpy as np

class EfficientPSO_DE_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, de_cr=0.5, de_f=0.8, w=0.5, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.de_f = de_f
        self.c1 = c1
        self.c2 = c2

    def __call__(self, func):
        def de_mutate(pop, target_idx):
            candidates = np.where(np.arange(len(pop)) != target_idx)[0]
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
                r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
                velocity = self.w * velocity + self.c1 * r1 * (pbest - swarm) + self.c2 * r2 * (gbest - swarm)
                swarm = np.clip(swarm + velocity, -5.0, 5.0)

                trial_vectors = np.apply_along_axis(de_mutate, 1, swarm, np.arange(len(swarm)))
                trial_fitness = np.array([func(v) for v in trial_vectors])
                
                improve_mask = trial_fitness < pbest_fitness
                pbest[improve_mask] = trial_vectors[improve_mask]
                pbest_fitness[improve_mask] = trial_fitness[improve_mask]
                
                gbest_idx = np.argmin(pbest_fitness)
                gbest = pbest[gbest_idx]

            return gbest

        return pso_de_optimize()
import numpy as np

class Improved_PSO_DE_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, de_cr=0.5, de_f=0.8, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.de_f = de_f
        self.c1 = c1
        self.c2 = c2

    def __call__(self, func):
        def de_mutate(swarm):
            choices = np.random.choice(swarm.shape[0], (3, swarm.shape[0]), replace=True)
            return np.clip(swarm + self.de_f * (swarm[choices[1]] - swarm[choices[2]]), -5.0, 5.0)

        def pso_de_optimize():
            swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
            pbest = swarm.copy()
            pbest_fitness = np.array([func(ind) for ind in swarm])
            gbest_idx = np.argmin(pbest_fitness)
            gbest = swarm[gbest_idx]
            inertia_weight = 0.9

            for _ in range(self.budget):
                velocity = inertia_weight * velocity + self.c1 * np.random.rand() * (pbest - swarm) + self.c2 * np.random.rand() * (gbest - swarm)
                swarm = np.clip(swarm + velocity, -5.0, 5.0)

                trial_vectors = de_mutate(swarm)
                trial_fitness = np.array([func(tv) for tv in trial_vectors])

                updates = trial_fitness < pbest_fitness
                pbest[updates] = trial_vectors[updates]
                pbest_fitness[updates] = trial_fitness[updates]

                gbest_idx = np.argmin(pbest_fitness)
                gbest = pbest[gbest_idx]
                
                inertia_weight = max(0.4, min(0.9, 0.9 - 0.5 * _ / self.budget))

            return gbest

        return pso_de_optimize()
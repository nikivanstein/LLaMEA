import numpy as np

class Enhanced_Improved_Vectorized_PSO_DE_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, de_cr=0.5, de_f=0.8, w=0.5, c1=1.5, c2=1.5):
        self.budget, self.dim, self.swarm_size, self.de_f, self.w, self.c1, self.c2 = budget, dim, swarm_size, de_f, w, c1, c2

    def __call__(self, func):
        def de_mutate(swarm):
            choices = np.random.choice(swarm.shape[0], (3, swarm.shape[0]), replace=True)
            return np.clip(swarm[choices[0]] + self.de_f * (swarm[choices[1]] - swarm[choices[2]], -5.0, 5.0)

        def pso_de_optimize():
            swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
            velocity, pbest, pbest_fitness = np.zeros_like(swarm), swarm.copy(), np.array([func(ind) for ind in swarm])
            gbest_idx = np.argmin(pbest_fitness)
            gbest, r1_r2 = swarm[gbest_idx], np.random.rand(2, self.swarm_size, self.dim)

            for _ in range(self.budget):
                velocity = self.w * velocity + self.c1 * r1_r2[0] * (pbest - swarm) + self.c2 * r1_r2[1] * (gbest - swarm)
                swarm = np.clip(swarm + velocity, -5.0, 5.0)

                trial_vectors = de_mutate(swarm)
                trial_fitness = np.array([func(tv) for tv in trial_vectors])

                updates = trial_fitness < pbest_fitness
                pbest[updates], pbest_fitness[updates] = trial_vectors[updates], trial_fitness[updates]

                gbest_idx = np.argmin(pbest_fitness)
                gbest = pbest[gbest_idx]

            return gbest

        return pso_de_optimize()
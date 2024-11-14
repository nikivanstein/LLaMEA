import numpy as np

class Enhanced_Vectorized_PSO_DE_Optimizer_Efficient:
    def __init__(self, budget, dim, swarm_size=30, de_cr=0.5, de_f=0.8, w=0.5, c1=1.5, c2=1.5):
        self.budget, self.dim, self.swarm_size, self.de_f, self.w, self.c1, self.c2 = budget, dim, swarm_size, de_f, w, c1, c2

    def __call__(self, func):
        def de_mutate(swarm):
            a_idx, b_idx, c_idx = np.random.choice(swarm.shape[0], (3, swarm.shape[0]), replace=True)
            return np.clip(swarm[a_idx] + self.de_f * (swarm[b_idx] - swarm[c_idx]), -5.0, 5.0)

        def pso_de_optimize():
            swarm, velocity = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim)), np.zeros((self.swarm_size, self.dim))
            pbest, pbest_fitness = swarm.copy(), np.array([func(ind) for ind in swarm])
            gbest_idx = np.argmin(pbest_fitness)
            gbest = swarm[gbest_idx]

            for _ in range(self.budget):
                r1_r2 = np.random.rand(2, self.swarm_size, self.dim)
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
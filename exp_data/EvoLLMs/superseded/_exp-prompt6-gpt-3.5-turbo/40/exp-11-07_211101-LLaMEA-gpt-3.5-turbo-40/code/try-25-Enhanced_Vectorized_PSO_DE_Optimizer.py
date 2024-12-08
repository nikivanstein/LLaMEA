import numpy as np

class Enhanced_Vectorized_PSO_DE_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, de_cr=0.5, de_f=0.8, w=0.5, c1=1.5, c2=1.5):
        self.budget, self.dim, self.swarm_size, self.de_cr, self.de_f, self.w, self.c1, self.c2 = budget, dim, swarm_size, de_cr, de_f, w, c1, c2

    def __call__(self, func):
        def de_mutate(pop, target_idx):
            candidates = np.delete(np.arange(len(pop)), target_idx)
            a, b, c = pop[np.random.choice(candidates, 3, replace=False)]
            return np.clip(a + self.de_f * (b - c), -5.0, 5.0)

        def pso_de_optimize():
            swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
            velocity = np.zeros((self.swarm_size, self.dim))
            pbest = swarm.copy()
            pbest_fitness = np.array(list(map(func, swarm)))  # Vectorized evaluation
            gbest_idx = np.argmin(pbest_fitness)
            gbest = swarm[gbest_idx]

            for _ in range(self.budget):
                r1_r2 = np.random.rand(2, self.swarm_size, self.dim)
                velocity = self.w * velocity + self.c1 * r1_r2[0] * (pbest - swarm) + self.c2 * r1_r2[1] * (gbest - swarm)
                swarm = np.clip(swarm + velocity, -5.0, 5.0)

                trial_vectors = np.array([de_mutate(swarm, i) for i in range(self.swarm_size)])
                trial_fitness = np.array(list(map(func, trial_vectors)))  # Vectorized evaluation

                updates = trial_fitness < pbest_fitness
                pbest_upd_indices = np.where(updates)[0]
                pbest[pbest_upd_indices] = trial_vectors[pbest_upd_indices]
                pbest_fitness[pbest_upd_indices] = trial_fitness[pbest_upd_indices]

                gbest_idx = np.argmin(pbest_fitness)
                gbest = pbest[gbest_idx]

            return gbest

        return pso_de_optimize()
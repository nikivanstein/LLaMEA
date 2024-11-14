import numpy as np
from joblib import Parallel, delayed

class Parallelized_Efficient_PSO_DE_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, de_cr=0.5, de_f=0.8, w=0.5, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.de_f = de_f
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def __call__(self, func):
        def de_mutate(swarm):
            a_idx, b_idx, c_idx = np.random.choice(swarm.shape[0], (3, swarm.shape[0]), replace=True)
            a, b, c = swarm[a_idx], swarm[b_idx], swarm[c_idx]
            return np.clip(a + self.de_f * (b - c), -5.0, 5.0)

        def parallel_fitness_evaluation(population):
            return np.array(Parallel(n_jobs=-1)(delayed(func)(ind) for ind in population))

        def parallel_update(pbest, swarm, trial_vectors, pbest_fitness, trial_fitness):
            updates = trial_fitness < pbest_fitness
            pbest_new = pbest.copy()
            pbest_new[updates] = trial_vectors[updates]
            pbest_fitness[updates] = trial_fitness[updates]
            gbest_idx = np.argmin(pbest_fitness)
            return pbest_new, gbest_idx

        swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        velocity = np.zeros((self.swarm_size, self.dim))
        pbest = swarm.copy()
        pbest_fitness = parallel_fitness_evaluation(swarm)
        gbest_idx = np.argmin(pbest_fitness)
        gbest = swarm[gbest_idx]

        for _ in range(self.budget):
            r1_r2 = np.random.rand(2, self.swarm_size, self.dim)
            velocity = self.w * velocity + self.c1 * r1_r2[0] * (pbest - swarm) + self.c2 * r1_r2[1] * (gbest - swarm)
            swarm = np.clip(swarm + velocity, -5.0, 5.0)

            trial_vectors = de_mutate(swarm)
            trial_fitness = parallel_fitness_evaluation(trial_vectors)

            pbest, gbest_idx = parallel_update(pbest, swarm, trial_vectors, pbest_fitness, trial_fitness)
            gbest = pbest[gbest_idx]

        return gbest
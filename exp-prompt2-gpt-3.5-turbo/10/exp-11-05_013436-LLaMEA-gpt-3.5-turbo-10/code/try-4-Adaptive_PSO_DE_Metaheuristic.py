import numpy as np

class Adaptive_PSO_DE_Metaheuristic(PSO_DE_Metaheuristic):
    def __init__(self, budget, dim, swarm_size=30, pso_w=0.5, pso_c1=1.5, pso_c2=1.5, de_f=0.5, de_cr=0.9):
        super().__init__(budget, dim, swarm_size, pso_w, pso_c1, pso_c2, de_f, de_cr)
        self.pso_w_range = [0.4, 0.9]
        self.pso_c1_range = [1.0, 2.0]
        self.pso_c2_range = [1.0, 2.0]
        self.de_f_range = [0.4, 0.9]
        self.de_cr_range = [0.7, 1.0]

    def __call__(self, func):
        def pso_de_optimizer():
            swarm = np.random.uniform(low=-5.0, high=5.0, size=(self.swarm_size, self.dim))
            velocities = np.zeros((self.swarm_size, self.dim))
            personal_best = swarm.copy()
            pbest_fitness = np.array([func(ind) for ind in swarm])
            gbest_fitness = np.min(pbest_fitness)
            gbest_idx = np.argmin(pbest_fitness)
            gbest = swarm[gbest_idx]

            for _ in range(self.budget):
                r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
                self.pso_w = np.clip(self.pso_w, *self.pso_w_range)
                self.pso_c1 = np.clip(self.pso_c1, *self.pso_c1_range)
                self.pso_c2 = np.clip(self.pso_c2, *self.pso_c2_range)
                self.de_f = np.clip(self.de_f, *self.de_f_range)
                self.de_cr = np.clip(self.de_cr, *self.de_cr_range)
                velocities = self.pso_w * velocities + self.pso_c1 * r1 * (personal_best - swarm) + self.pso_c2 * r2 * (gbest - swarm)
                swarm = swarm + velocities

                for i in range(self.swarm_size):
                    trial = swarm[i].copy()
                    idxs = list(range(self.swarm_size))
                    idxs.remove(i)
                    a, b, c = swarm[np.random.choice(idxs, 3, replace=False)]
                    j_rand = np.random.randint(0, self.dim)
                    for j in range(self.dim):
                        if np.random.rand() < self.de_cr or j == j_rand:
                            trial[j] = a[j] + self.de_f * (b[j] - c[j])
                    trial_fitness = func(trial)
                    if trial_fitness < pbest_fitness[i]:
                        pbest_fitness[i] = trial_fitness
                        personal_best[i] = trial
                        if trial_fitness < gbest_fitness:
                            gbest_fitness = trial_fitness
                            gbest = trial

            return gbest, gbest_fitness

        return pso_de_optimizer()
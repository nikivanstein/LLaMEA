import numpy as np

class Enhanced_Adaptive_Dynamic_Mutation_Population_Resizing(Enhanced_Dynamic_Mutation_Population_Resizing):
    def __init__(self, budget, dim, swarm_size=30, pso_w=0.5, pso_c1=1.5, pso_c2=1.5, de_f=0.5, de_cr=0.9, mutation_prob=0.1, mutation_scale=0.1):
        super().__init__(budget, dim, swarm_size, pso_w, pso_c1, pso_c2, de_f, de_cr, mutation_prob, mutation_scale)
        self.dynamic_mutation_scale = mutation_scale

    def __call__(self, func):
        def pso_de_optimizer():
            swarm_size = self.swarm_size
            swarm = np.random.uniform(low=-5.0, high=5.0, size=(swarm_size, self.dim))
            swarm += self.dynamic_mutation_scale * np.random.uniform(low=-1.0, high=1.0, size=(swarm_size, self.dim))
            velocities = np.zeros((swarm_size, self.dim))
            personal_best = swarm.copy()
            pbest_fitness = np.array([func(ind) for ind in swarm])
            gbest_fitness = np.min(pbest_fitness)
            gbest_idx = np.argmin(pbest_fitness)
            gbest = swarm[gbest_idx]

            for iter_count in range(self.budget):
                progress = iter_count / self.initial_budget
                dynamic_params = [(1 - progress) * p + progress * p * self.dynamic_threshold for p in [self.pso_w, self.pso_c1, self.pso_c2, self.de_f, self.de_cr]]

                r1, r2 = np.random.rand(swarm_size, self.dim), np.random.rand(swarm_size, self.dim)
                velocities = dynamic_params[0] * velocities + dynamic_params[1] * r1 * (personal_best - swarm) + dynamic_params[2] * r2 * (gbest - swarm)
                swarm = swarm + velocities

                swarm_size = max(self.min_swarm_size, min(self.max_swarm_size, int(self.swarm_size * (1 - progress))))
                if swarm_size != self.swarm_size:
                    self.swarm_size = swarm_size
                    velocities = velocities[:swarm_size]
                    swarm = swarm[:swarm_size]
                    personal_best = personal_best[:swarm_size]
                    pbest_fitness = pbest_fitness[:swarm_size]
                
                if np.random.rand() < 0.1:  # 10% rate of change
                    self.dynamic_mutation_scale *= (1 - progress)  # Adjust mutation scale based on progress

                for i in range(swarm_size):
                    trial = swarm[i].copy()
                    idxs = list(range(swarm_size))
                    idxs.remove(i)
                    a, b, c = swarm[np.random.choice(idxs, 3, replace=False)]
                    j_rand = np.random.randint(0, self.dim)
                    for j in range(self.dim):
                        if np.random.rand() < dynamic_params[4] or j == j_rand:
                            if np.random.rand() < (self.mutation_prob * (1 - progress)):
                                trial[j] = np.random.uniform(low=-5.0, high=5.0)
                            else:
                                beta = np.random.normal(0, 1, 1)[0]
                                trial[j] = a[j] + beta * (b[j] - c[j])
                    trial_fitness = func(trial)
                    if trial_fitness < pbest_fitness[i]:
                        pbest_fitness[i] = trial_fitness
                        personal_best[i] = trial
                        if trial_fitness < gbest_fitness:
                            gbest_fitness = trial_fitness
                            gbest = trial

            return gbest, gbest_fitness

        return pso_de_optimizer()
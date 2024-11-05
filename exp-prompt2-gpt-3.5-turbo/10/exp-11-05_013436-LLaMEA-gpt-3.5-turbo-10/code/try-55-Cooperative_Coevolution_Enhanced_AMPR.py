import numpy as np

class Cooperative_Coevolution_Enhanced_AMPR(Enhanced_Adaptive_Mutation_Population_Resizing):
    def __init__(self, budget, dim, num_subpopulations=5, subpopulation_size=10, pso_w=0.5, pso_c1=1.5, pso_c2=1.5, de_f=0.5, de_cr=0.9, mutation_prob=0.1):
        super().__init__(budget, dim, num_subpopulations * subpopulation_size, pso_w, pso_c1, pso_c2, de_f, de_cr, mutation_prob)
        self.num_subpopulations = num_subpopulations
        self.subpopulation_size = subpopulation_size
    
    def __call__(self, func):
        def pso_de_cc_optimizer():
            swarm = np.random.uniform(low=-5.0, high=5.0, size=(self.num_subpopulations * self.subpopulation_size, self.dim))
            velocities = np.zeros((self.num_subpopulations * self.subpopulation_size, self.dim))
            personal_best = swarm.copy()
            pbest_fitness = np.array([func(ind) for ind in swarm])
            gbest_fitness = np.min(pbest_fitness)
            gbest_idx = np.argmin(pbest_fitness)
            gbest = swarm[gbest_idx]

            for iter_count in range(self.budget):
                progress = iter_count / self.initial_budget
                dynamic_params = [(1 - progress) * p + progress * p * self.dynamic_threshold for p in [self.pso_w, self.pso_c1, self.pso_c2, self.de_f, self.de_cr]]

                r1, r2 = np.random.rand(self.num_subpopulations * self.subpopulation_size, self.dim), np.random.rand(self.num_subpopulations * self.subpopulation_size, self.dim)
                velocities = dynamic_params[0] * velocities + dynamic_params[1] * r1 * (personal_best - swarm) + dynamic_params[2] * r2 * (gbest - swarm)
                swarm = swarm + velocities

                # Dynamic Swarm Size Adjustment
                swarm_size = max(self.min_swarm_size, min(self.max_swarm_size, int(self.swarm_size * (1 - progress))))
                if swarm_size != self.swarm_size:
                    self.swarm_size = swarm_size
                    velocities = velocities[:swarm_size]
                    swarm = swarm[:swarm_size]
                    personal_best = personal_best[:swarm_size]
                    pbest_fitness = pbest_fitness[:swarm_size]

                for i in range(self.num_subpopulations):
                    subpop = swarm[i * self.subpopulation_size: (i + 1) * self.subpopulation_size]
                    subpop_best_idx = np.argmin(pbest_fitness[i * self.subpopulation_size: (i + 1) * self.subpopulation_size])
                    subpop_best = personal_best[subpop_best_idx]
                    for j in range(self.subpopulation_size):
                        subpop[j] = subpop_best  # Cooperative Coevolution Strategy

                    for j in range(self.subpopulation_size):
                        trial = subpop[j].copy()
                        idxs = list(range(self.num_subpopulations * self.subpopulation_size))
                        idxs.remove(i * self.subpopulation_size + j)
                        a, b, c = swarm[np.random.choice(idxs, 3, replace=False)]
                        j_rand = np.random.randint(0, self.dim)
                        for k in range(self.dim):
                            if np.random.rand() < dynamic_params[4] or k == j_rand:
                                if np.random.rand() < self.mutation_prob:
                                    trial[k] = np.random.uniform(low=-5.0, high=5.0)
                                else:
                                    beta = np.random.normal(0, 1, 1)[0]
                                    trial[k] = a[k] + beta * (b[k] - c[k])
                        trial_fitness = func(trial)
                        if trial_fitness < pbest_fitness[i * self.subpopulation_size + j]:
                            pbest_fitness[i * self.subpopulation_size + j] = trial_fitness
                            personal_best[i * self.subpopulation_size + j] = trial
                            if trial_fitness < gbest_fitness:
                                gbest_fitness = trial_fitness
                                gbest = trial

            return gbest, gbest_fitness

        return pso_de_cc_optimizer()
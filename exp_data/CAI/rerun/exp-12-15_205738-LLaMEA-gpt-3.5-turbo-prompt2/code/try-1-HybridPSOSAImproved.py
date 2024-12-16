class HybridPSOSAImproved(HybridPSOSA):
    def __init__(self, budget, dim, num_particles=30, max_iterations=1000, c1=1.5, c2=1.5, initial_temp=100, cooling_rate=0.95, inertia_min=0.4, inertia_max=0.9):
        super().__init__(budget, dim, num_particles, max_iterations, c1, c2, initial_temp, cooling_rate)
        self.inertia_min = inertia_min
        self.inertia_max = inertia_max
        
    def __call__(self, func):
        lb, ub = -5.0, 5.0
        swarm = lb + (ub - lb) * np.random.rand(self.num_particles, self.dim)
        velocity = np.zeros((self.num_particles, self.dim))
        personal_best_pos = swarm.copy()
        personal_best_val = np.full(self.num_particles, np.inf)
        global_best_pos = np.zeros(self.dim)
        global_best_val = np.inf

        curr_temp = self.initial_temp

        for _ in range(self.max_iterations):
            inertia_weight = self.inertia_max - ((_ + 1) / self.max_iterations) * (self.inertia_max - self.inertia_min)
            for i in range(self.num_particles):
                fitness = func(swarm[i])
                if fitness < personal_best_val[i]:
                    personal_best_val[i] = fitness
                    personal_best_pos[i] = swarm[i].copy()
                    if fitness < global_best_val:
                        global_best_val = fitness
                        global_best_pos = swarm[i].copy()

                r1, r2 = np.random.rand(), np.random.rand()
                velocity[i] = inertia_weight * velocity[i] + self.c1 * r1 * (personal_best_pos[i] - swarm[i]) + self.c2 * r2 * (global_best_pos - swarm[i])
                swarm[i] = np.clip(swarm[i] + velocity[i], lb, ub)

                new_pos = swarm[i] + np.random.normal(0, curr_temp, self.dim)
                new_pos = np.clip(new_pos, lb, ub)
                new_fitness = func(new_pos)

                if new_fitness < fitness or np.random.rand() < np.exp((fitness - new_fitness) / curr_temp):
                    swarm[i] = new_pos.copy()

            curr_temp *= self.cooling_rate

        return global_best_val
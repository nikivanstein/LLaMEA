class ImprovedDynamicHybridPSOSA_DE(DynamicHybridPSOSA_DE):
    def __init__(self, budget, dim, num_particles=30, max_iterations=1000, c1=1.5, c2=1.5, initial_temp=100, cooling_rate=0.95, f=0.5, cr=0.9, f_lower=0.1, f_upper=0.9, cr_lower=0.1, cr_upper=0.9):
        super().__init__(budget, dim, num_particles, max_iterations, c1, c2, initial_temp, cooling_rate, f, cr, f_lower, f_upper, cr_lower, cr_upper)
    
    def dynamic_population(self, swarm, personal_best_pos, personal_best_val):
        if np.random.rand() < 0.1:  # 2% change - randomly remove a particle
            remove_idx = np.random.randint(self.num_particles)
            swarm = np.delete(swarm, remove_idx, axis=0)
            personal_best_pos = np.delete(personal_best_pos, remove_idx, axis=0)
            personal_best_val = np.delete(personal_best_val, remove_idx)
            
        if np.random.rand() < 0.1:  # 2% change - randomly add a particle
            new_particle = -5.0 + 10.0 * np.random.rand(self.dim)
            swarm = np.vstack([swarm, new_particle])
            personal_best_pos = np.vstack([personal_best_pos, new_particle])
            personal_best_val = np.append(personal_best_val, np.inf)
        
        return swarm, personal_best_pos, personal_best_val
    
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
            swarm, personal_best_pos, personal_best_val = self.dynamic_population(swarm, personal_best_pos, personal_best_val)  # Dynamic population

            for i in range(len(swarm)):
                fitness = func(swarm[i])
                if fitness < personal_best_val[i]:
                    personal_best_val[i] = fitness
                    personal_best_pos[i] = swarm[i].copy()
                    if fitness < global_best_val:
                        global_best_val = fitness
                        global_best_pos = swarm[i].copy()

                r1, r2 = np.random.rand(), np.random.rand()
                velocity[i] = 0.5 * velocity[i] + self.c1 * r1 * (personal_best_pos[i] - swarm[i]) + self.c2 * r2 * (global_best_pos - swarm[i])
                swarm[i] = np.clip(swarm[i] + velocity[i], lb, ub)

                a, b, c = swarm[np.random.choice(range(len(swarm)), 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < self.cr
                trial = swarm[i].copy()
                trial[crossover] = mutant[crossover]
                trial_fitness = func(trial)

                if trial_fitness < fitness:
                    swarm[i] = trial.copy()

            curr_temp *= self.cooling_rate

            self.f = max(self.f_lower, min(self.f_upper, self.chaotic_map(self.f)))  # Dynamic F using chaotic map
            self.cr = max(self.cr_lower, min(self.cr_upper, self.chaotic_map(self.cr)))  # Dynamic CR using chaotic map

        return global_best_val
class EnhancedDynamicHarmonySearchRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.par_min = 0.1
        self.par_max = 0.9
        self.bandwidth_min = 0.01
        self.bandwidth_max = 0.1
        self.mutation_factor_min = 0.1
        self.mutation_factor_max = 0.9
        self.crossover_prob_min = 0.5
        self.crossover_prob_max = 0.9
        self.inertia_weight = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.pso_w = 0.5
        self.pso_c1 = 1.5
        self.pso_c2 = 1.5
        self.ga_mut_prob = 0.1
        self.sa_iter = 100
        self.sa_T0 = 100
        self.sa_Tf = 0.001

    def __call__(self, func):
        def update_parameters(iteration):
            par = self.par_min + (self.par_max - self.par_min) * (iteration / self.budget)
            mutation_factor = self.mutation_factor_min + (self.mutation_factor_max - self.mutation_factor_min) * (iteration / self.budget)
            crossover_prob = self.crossover_prob_min + (self.crossover_prob_max - self.crossover_prob_min) * (iteration / self.budget)
            return par, mutation_factor, crossover_prob

        def update_velocity_position(particle, particle_best, global_best, mutation_factor, crossover_prob):
            velocity = self.inertia_weight * particle['velocity'] + self.c1 * np.random.rand() * (particle_best - particle['position']) + self.c2 * np.random.rand() * (global_best - particle['position'])
            position = particle['position'] + mutation_factor * np.random.rand() * (particle_best - particle['position']) + crossover_prob * np.random.rand() * (global_best - particle['position'])
            position = np.clip(position, -5.0, 5.0)
            return position, velocity

        harmony_memory = initialize_harmony_memory()
        particles = [{'position': np.random.uniform(-5.0, 5.0, self.dim), 'velocity': np.zeros(self.dim)} for _ in range(self.harmony_memory_size)]
        best_solution = None
        best_fitness = np.inf

        for iteration in range(self.budget):
            par, mutation_factor, crossover_prob = update_parameters(iteration)

            for idx, particle in enumerate(particles):
                new_position, new_velocity = update_velocity_position(particle, best_solution, best_solution, mutation_factor, crossover_prob)
                particles[idx]['position'] = new_position
                particles[idx]['velocity'] = new_velocity

            idx = np.argmax([func(h) for h in harmony_memory])
            if new_fitness < func(harmony_memory[idx]):
                harmony_memory[idx] = new_harmony

        return best_solution
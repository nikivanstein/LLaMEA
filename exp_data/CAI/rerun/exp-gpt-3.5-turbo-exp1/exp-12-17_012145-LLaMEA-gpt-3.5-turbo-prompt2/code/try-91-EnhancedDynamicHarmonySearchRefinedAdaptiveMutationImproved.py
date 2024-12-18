class EnhancedDynamicHarmonySearchRefinedAdaptiveMutationImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 15
        self.par_min = 0.1
        self.par_max = 0.9
        self.bandwidth_min = 0.01
        self.bandwidth_max = 0.1
        self.mutation_factor_min = 0.1
        self.mutation_factor_max = 0.9
        self.crossover_prob = 0.7
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
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))

        def update_parameters(iteration):
            par = self.par_min + (self.par_max - self.par_min) * (iteration / self.budget)
            mutation_factor = self.mutation_factor_min + (self.mutation_factor_max - self.mutation_factor_min) * (iteration / self.budget)
            return par, mutation_factor

        def improvise_harmony(harmony_memory, par):
            new_harmony = np.copy(harmony_memory[np.random.choice(range(self.harmony_memory_size))])
            for i in range(self.dim):
                if np.random.uniform() < par:
                    new_harmony[i] += np.random.uniform(-0.1, 0.1)
                    new_harmony[i] = np.clip(new_harmony[i], -5.0, 5.0)
            return new_harmony

        def update_velocity_position(particle, particle_best, global_best):
            velocity = self.inertia_weight * particle['velocity'] + self.c1 * np.random.rand() * (particle_best - particle['position']) + self.c2 * np.random.rand() * (global_best - particle['position'])
            position = particle['position'] + velocity
            position = np.clip(position, -5.0, 5.0)
            return position, velocity

        harmony_memory = initialize_harmony_memory()
        particles = [{'position': np.random.uniform(-5.0, 5.0, self.dim), 'velocity': np.zeros(self.dim)} for _ in range(self.harmony_memory_size)]
        best_solution = None
        best_fitness = np.inf

        for iteration in range(self.budget):
            par, mutation_factor = update_parameters(iteration)
            new_harmony = improvise_harmony(harmony_memory, par)

            de_harmony = differential_evolution(func, bounds=[(-5.0, 5.0)]*self.dim, mutation=mutation_factor, recombination=self.crossover_prob).x

            ps = minimize(func, new_harmony, method='Powell').x

            sa = dual_annealing(func, bounds=[(-5.0, 5.0)]*self.dim, maxiter=self.sa_iter, initial_temp=self.sa_T0, restart_temp_ratio=self.sa_Tf).x

            basinhop = basinhopping(func, new_harmony, stepsize=0.5).x

            new_fitness = func(new_harmony)
            de_fitness = func(de_harmony)
            ps_fitness = func(ps)
            sa_fitness = func(sa)
            basinhop_fitness = func(basinhop)

            if new_fitness < best_fitness:
                best_solution = new_harmony
                best_fitness = new_fitness

            if de_fitness < best_fitness:
                best_solution = de_harmony
                best_fitness = de_fitness

            if ps_fitness < best_fitness:
                best_solution = ps
                best_fitness = ps_fitness

            if sa_fitness < best_fitness:
                best_solution = sa
                best_fitness = sa_fitness

            if basinhop_fitness < best_fitness:
                best_solution = basinhop
                best_fitness = basinhop_fitness

            for idx, particle in enumerate(particles):
                new_position, new_velocity = update_velocity_position(particle, best_solution, best_solution)
                particles[idx]['position'] = new_position
                particles[idx]['velocity'] = new_velocity

            idx = np.argmax([func(h) for h in harmony_memory])
            if new_fitness < func(harmony_memory[idx]):
                harmony_memory[idx] = new_harmony

        return best_solution
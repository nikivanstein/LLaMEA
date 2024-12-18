class EnhancedDynamicHarmonySearchRefinedHybrid:
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
        self.crossover_prob = 0.7
        self.inertia_weight = 0.7  # Add dynamic inertia weight
        self.c1 = 1.5
        self.c2 = 1.5
        self.pso_w = 0.5
        self.pso_c1 = 1.5
        self.pso_c2 = 1.5
        self.ga_mut_prob = 0.1

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))

        def update_parameters(iteration):
            par = self.par_min + (self.par_max - self.par_min) * (iteration / self.budget)
            bandwidth = self.bandwidth_min + (self.bandwidth_max - self.bandwidth_min) * (iteration / self.budget)
            mutation_factor = self.mutation_factor_min + (self.mutation_factor_max - self.mutation_factor_min) * (iteration / self.budget)
            inertia_weight = self.inertia_weight - (self.inertia_weight / self.budget) * iteration  # Update inertia weight dynamically
            return par, bandwidth, mutation_factor, inertia_weight

        def improvise_harmony(harmony_memory, par, bandwidth):
            new_harmony = np.copy(harmony_memory[np.random.randint(0, self.harmony_memory_size)])
            for i in range(self.dim):
                if np.random.uniform() < par:
                    new_harmony[i] += np.random.uniform(-bandwidth, bandwidth)
                    new_harmony[i] = np.clip(new_harmony[i], -5.0, 5.0)
            return new_harmony

        harmony_memory = initialize_harmony_memory()
        best_solution = None
        best_fitness = np.inf

        for iteration in range(self.budget):
            par, bandwidth, mutation_factor, inertia_weight = update_parameters(iteration)
            new_harmony = improvise_harmony(harmony_memory, par, bandwidth)

            de_harmony = differential_evolution(func, bounds=[(-5.0, 5.0)]*self.dim, mutation=mutation_factor, recombination=self.crossover_prob).x
            sa = dual_annealing(func, bounds=[(-5.0, 5.0)]*self.dim).x

            new_fitness = func(new_harmony)
            de_fitness = func(de_harmony)
            sa_fitness = func(sa)

            if new_fitness < best_fitness:
                best_solution = new_harmony
                best_fitness = new_fitness

            if de_fitness < best_fitness:
                best_solution = de_harmony
                best_fitness = de_fitness

            if sa_fitness < best_fitness:
                best_solution = sa
                best_fitness = sa_fitness

            idx = np.argmax([func(h) for h in harmony_memory])
            if new_fitness < func(harmony_memory[idx]):
                harmony_memory[idx] = new_harmony

        return best_solution
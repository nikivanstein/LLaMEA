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
        self.crossover_prob = 0.7
        self.inertia_weight = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.pso_w = 0.5
        self.pso_c1 = 1.5
        self.pso_c2 = 1.5

    def __call__(self, func):
        def update_parameters(iteration):
            par = self.par_min + (self.par_max - self.par_min) * (iteration / self.budget)
            bandwidth = self.bandwidth_min + (self.bandwidth_max - self.bandwidth_min) * (iteration / self.budget)
            mutation_factor = self.mutation_factor_min + (self.mutation_factor_max - self.mutation_factor_min) * (iteration / self.budget)
            return par, bandwidth, mutation_factor

        def improvise_harmony(harmony_memory, par, bandwidth):
            new_harmony = np.copy(harmony_memory[np.random.randint(0, self.harmony_memory_size)])
            for i in range(self.dim):
                if np.random.uniform() < par:
                    new_harmony[i] += np.random.uniform(-bandwidth, bandwidth)
                    new_harmony[i] = np.clip(new_harmony[i], -5.0, 5.0)
            return new_harmony

        def update_mutate(new_harmony, mutation_factor):
            mutated_harmony = new_harmony + mutation_factor * np.random.normal(0, 1, self.dim)
            mutated_harmony = np.clip(mutated_harmony, -5.0, 5.0)
            return mutated_harmony

        harmony_memory = initialize_harmony_memory()
        particles = [{'position': np.random.uniform(-5.0, 5.0, self.dim), 'velocity': np.zeros(self.dim)} for _ in range(self.harmony_memory_size)]
        best_solution = None
        best_fitness = np.inf

        for iteration in range(self.budget):
            par, bandwidth, mutation_factor = update_parameters(iteration)
            new_harmony = improvise_harmony(harmony_memory, par, bandwidth)
            mutated_harmony = update_mutate(new_harmony, mutation_factor)

            # Other parts of the algorithm remain unchanged

        return best_solution
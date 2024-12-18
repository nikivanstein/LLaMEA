class ImprovedDynamicHarmonySearch(DynamicHarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.explore_prob = 0.2

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))

        def update_parameters(iteration):
            par = self.par_min + (self.par_max - self.par_min) * (iteration / self.budget)
            bandwidth = self.bandwidth_min + (self.bandwidth_max - self.bandwidth_min) * (iteration / self.budget)
            return par, bandwidth

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
            par, bandwidth = update_parameters(iteration)
            new_harmony = improvise_harmony(harmony_memory, par, bandwidth)
            new_fitness = func(new_harmony)
            if new_fitness < best_fitness:
                best_solution = new_harmony
                best_fitness = new_fitness

            idx = np.argmax([func(h) for h in harmony_memory])
            if new_fitness < func(harmony_memory[idx]):
                harmony_memory[idx] = new_harmony

            if np.random.uniform() < self.explore_prob:
                harmony_memory = np.vstack([harmony_memory, new_harmony])
                self.harmony_memory_size += 1

        return best_solution
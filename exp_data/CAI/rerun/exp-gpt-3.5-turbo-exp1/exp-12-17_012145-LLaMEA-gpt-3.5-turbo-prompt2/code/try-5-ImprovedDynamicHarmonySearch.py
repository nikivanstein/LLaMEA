class ImprovedDynamicHarmonySearch(DynamicHarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def opposition_based_improvisation(self, harmony):
        opp_harmony = -1 * harmony
        return opp_harmony

    def __call__(self, func):
        def initialize_harmony_memory():
            harmony_memory = np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))
            opp_harmony_memory = np.array([self.opposition_based_improvisation(h) for h in harmony_memory])
            return np.vstack((harmony_memory, opp_harmony_memory))

        def improvise_harmony(harmony_memory, par, bandwidth):
            new_harmony = np.copy(harmony_memory[np.random.randint(0, 2*self.harmony_memory_size)])
            for i in range(self.dim):
                if np.random.uniform() < par:
                    new_harmony[i] += np.random.uniform(-bandwidth, bandwidth)
                    new_harmony[i] = np.clip(new_harmony[i], -5.0, 5.0)
            return new_harmony

        harmony_memory = initialize_harmony_memory()
        best_solution = None
        best_fitness = np.inf

        for iteration in range(self.budget):
            par, bandwidth = self.update_parameters(iteration)
            new_harmony = improvise_harmony(harmony_memory, par, bandwidth)
            new_fitness = func(new_harmony)
            if new_fitness < best_fitness:
                best_solution = new_harmony
                best_fitness = new_fitness

            idx = np.argmax([func(h) for h in harmony_memory])
            if new_fitness < func(harmony_memory[idx]):
                harmony_memory[idx] = new_harmony

        return best_solution
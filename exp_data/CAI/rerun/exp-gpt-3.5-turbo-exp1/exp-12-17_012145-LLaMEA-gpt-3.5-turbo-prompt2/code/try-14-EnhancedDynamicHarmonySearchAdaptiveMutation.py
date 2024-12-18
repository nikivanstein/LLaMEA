class EnhancedDynamicHarmonySearchAdaptiveMutation(EnhancedDynamicHarmonySearchRefined):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_factors = np.linspace(0.1, 0.9, self.budget)

    def __call__(self, func):
        harmony_memory = self.initialize_harmony_memory()
        best_solution = None
        best_fitness = np.inf

        for iteration in range(self.budget):
            par, bandwidth = self.update_parameters(iteration)
            mutation_factor = self.mutation_factors[iteration]
            new_harmony = self.improvise_harmony(harmony_memory, par, bandwidth)
            new_harmony_opposite = self.apply_opposition(new_harmony)
            
            de_harmony = self.differential_evolution(harmony_memory, mutation_factor, self.crossover_prob)
            de_harmony_opposite = self.apply_opposition(de_harmony)

            new_fitness = func(new_harmony)
            new_fitness_opposite = func(new_harmony_opposite)
            de_fitness = func(de_harmony)
            de_fitness_opposite = func(de_harmony_opposite)

            if new_fitness < best_fitness:
                best_solution = new_harmony
                best_fitness = new_fitness

            if new_fitness_opposite < best_fitness:
                best_solution = new_harmony_opposite
                best_fitness = new_fitness_opposite

            if de_fitness < best_fitness:
                best_solution = de_harmony
                best_fitness = de_fitness

            if de_fitness_opposite < best_fitness:
                best_solution = de_harmony_opposite
                best_fitness = de_fitness_opposite

            idx = np.argmax([func(h) for h in harmony_memory])
            if new_fitness < func(harmony_memory[idx]):
                harmony_memory[idx] = new_harmony

        return best_solution
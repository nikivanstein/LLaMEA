class AdaptiveDynamicHarmonySearchRefined(EnhancedDynamicHarmonySearchRefined):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.gradient_threshold = 0.5

    def adaptive_mutation(self, harmony_memory, best_fitness, new_fitness, iteration):
        gradient = (best_fitness - new_fitness) / (self.budget - iteration)
        if gradient > self.gradient_threshold:
            self.mutation_factor *= 1.1
        else:
            self.mutation_factor *= 0.9

    def __call__(self, func):
        # Existing code remains the same until the iteration loop
        for iteration in range(self.budget):
            par, bandwidth = update_parameters(iteration)
            new_harmony = improvise_harmony(harmony_memory, par, bandwidth)
            new_harmony_opposite = apply_opposition(new_harmony)
            
            de_harmony = differential_evolution(harmony_memory, self.mutation_factor, self.crossover_prob)
            de_harmony_opposite = apply_opposition(de_harmony)

            new_fitness = func(new_harmony)
            new_fitness_opposite = func(new_harmony_opposite)
            de_fitness = func(de_harmony)
            de_fitness_opposite = func(de_harmony_opposite)

            adaptive_mutation(harmony_memory, best_fitness, new_fitness, iteration)
            
            # Rest of the existing code for updating best solution and memory
            
        return best_solution
class MultiPopulationDynamicPitchHybridHarmonySearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def generate_harmony():
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        def improvise(harmony_memory, pitch_adjust_rate):
            new_harmony = np.copy(harmony_memory[np.random.randint(len(harmony_memory))])
            for i in range(self.dim):
                if np.random.rand() < pitch_adjust_rate:
                    new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
            return new_harmony

        def differential_evolution(harmony_memory, fitness_values):
            mutation_factor = 0.5
            crossover_rate = 0.9
            for i in range(len(harmony_memory)):
                target_idx = np.random.choice(list(set(range(len(harmony_memory))) - {i}))
                base, target = harmony_memory[i], harmony_memory[target_idx]
                donor = base + mutation_factor * (target - harmony_memory[np.random.choice(range(len(harmony_memory)))])

                trial = np.copy(base)
                for j in range(len(trial)):
                    if np.random.rand() < crossover_rate:
                        trial[j] = donor[j] if np.random.rand() < 0.5 else base[j]

                trial_fitness = func(trial)
                if trial_fitness < fitness_values[i]:
                    harmony_memory[i] = trial
                    fitness_values[i] = trial_fitness

        num_populations = 5
        harmony_memory_size = 10
        pitch_adjust_rate = 0.1
        populations = [np.array([generate_harmony() for _ in range(harmony_memory_size)]) for _ in range(num_populations)]
        fitness_values = [np.array([func(harmony) for harmony in population]) for population in populations]

        for _ in range(self.budget - harmony_memory_size):
            for i in range(num_populations):
                new_harmony = improvise(populations[i], pitch_adjust_rate)
                new_fitness = func(new_harmony)
                if new_fitness < np.max(fitness_values[i]):
                    index = np.argmax(fitness_values[i])
                    populations[i][index] = new_harmony
                    fitness_values[i][index] = new_fitness
                differential_evolution(populations[i], fitness_values[i])

            all_harmonies = np.concatenate(populations)
            all_fitnesses = np.concatenate(fitness_values)

            local_search_indices = np.argsort(all_fitnesses)[:harmony_memory_size]
            for idx, harmony_idx in enumerate(local_search_indices):
                population_idx = harmony_idx // harmony_memory_size
                local_search_harmony = minimize(func, all_harmonies[harmony_idx], method='Nelder-Mead').x
                local_search_fitness = func(local_search_harmony)
                if local_search_fitness < fitness_values[population_idx][harmony_idx % harmony_memory_size]:
                    populations[population_idx][harmony_idx % harmony_memory_size] = local_search_harmony
                    fitness_values[population_idx][harmony_idx % harmony_memory_size] = local_search_fitness

            best_fitness = min(all_fitnesses)
            pitch_adjust_rate = max(0.01, min(0.5, pitch_adjust_rate + 0.1 * (all_fitnesses.sum() - best_fitness * len(all_fitnesses)))

        best_index = np.argmin(all_fitnesses)
        return all_harmonies[best_index]
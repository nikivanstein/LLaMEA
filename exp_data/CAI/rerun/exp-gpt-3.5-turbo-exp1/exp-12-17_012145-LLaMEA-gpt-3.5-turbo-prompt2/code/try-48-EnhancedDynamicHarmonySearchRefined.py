import numpy as np

class EnhancedDynamicHarmonySearchRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.par_min = 0.1
        self.par_max = 0.9
        self.bandwidth_min = 0.01
        self.bandwidth_max = 0.1
        self.mutation_factor = 0.5
        self.crossover_prob = 0.7
        self.inertia_weight = 0.7
        self.c1 = 1.5
        self.c2 = 1.5

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

        def apply_opposition(new_harmony):
            return 2.0 * np.mean(new_harmony) - new_harmony

        def update_velocity_position(particle, particle_best, global_best):
            velocity = self.inertia_weight * particle['velocity'] + self.c1 * np.random.rand() * (particle_best - particle['position']) + self.c2 * np.random.rand() * (global_best - particle['position'])
            position = particle['position'] + velocity
            position = np.clip(position, -5.0, 5.0)
            return position, velocity

        harmony_memory = initialize_harmony_memory()
        best_solution = None
        best_fitness = np.inf

        for iteration in range(self.budget):
            par, bandwidth = update_parameters(iteration)
            new_harmony = improvise_harmony(harmony_memory, par, bandwidth)
            new_harmony_opposite = apply_opposition(new_harmony)
    
            de_harmony = np.array([differential_evolution(harmony_memory, self.mutation_factor, self.crossover_prob) for _ in range(self.harmony_memory_size)])
            de_harmony_opposite = apply_opposition(de_harmony)
    
            new_fitness = np.array([func(h) for h in new_harmony])
            new_fitness_opposite = np.array([func(h) for h in new_harmony_opposite])
            de_fitness = np.array([func(h) for h in de_harmony])
            de_fitness_opposite = np.array([func(h) for h in de_harmony_opposite])
    
            harmony_memory = np.array([new_harmony if nf < f else h for h, nf, f in zip(harmony_memory, new_fitness, func(harmony_memory))])
            harmony_memory = np.array([new_harmony_opposite if nf < f else h for h, nf, f in zip(harmony_memory, new_fitness_opposite, func(harmony_memory))])
            harmony_memory = np.array([dh if df < f else h for h, dh, df, f in zip(harmony_memory, de_harmony, de_fitness, func(harmony_memory))])
            harmony_memory = np.array([de_h if df < f else h for h, de_h, df, f in zip(harmony_memory, de_harmony_opposite, de_fitness_opposite, func(harmony_memory))])
    
            idx = np.argmin([func(h) for h in harmony_memory])
            best_solution = harmony_memory[idx]
            best_fitness = func(best_solution)
    
        return best_solution

        def differential_evolution(harmony_memory, F, CR):
            r1, r2, r3 = np.random.choice(range(len(harmony_memory)), 3, replace=False)
            mutant_vector = harmony_memory[r1] + F * (harmony_memory[r2] - harmony_memory[r3])
            crossover_mask = np.random.rand(self.dim) < CR
            trial_vector = np.where(crossover_mask, mutant_vector, harmony_memory[np.random.randint(len(harmony_memory))])
            return trial_vector
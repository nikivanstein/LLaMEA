import numpy as np
from scipy.optimize import differential_evolution

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
        self.pso_c1 = 1.5
        self.pso_c2 = 1.5
        self.pso_inertia_weight = 0.7

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
        particles = [{'position': np.random.uniform(-5.0, 5.0, self.dim), 'velocity': np.zeros(self.dim)} for _ in range(self.harmony_memory_size)]
        best_solution = None
        best_fitness = np.inf

        for iteration in range(self.budget):
            par, bandwidth = update_parameters(iteration)
            new_harmony = improvise_harmony(harmony_memory, par, bandwidth)
            new_harmony_opposite = apply_opposition(new_harmony)
            
            de_harmony = differential_evolution(func, bounds=[(-5.0, 5.0)]*self.dim).x
            de_harmony_opposite = apply_opposition(de_harmony)

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

            for idx, particle in enumerate(particles):
                new_position, new_velocity = update_velocity_position(particle, best_solution, best_solution)
                particles[idx]['position'] = new_position
                particles[idx]['velocity'] = new_velocity

            idx = np.argmax([func(h) for h in harmony_memory])
            if new_fitness < func(harmony_memory[idx]):
                harmony_memory[idx] = new_harmony

        return best_solution
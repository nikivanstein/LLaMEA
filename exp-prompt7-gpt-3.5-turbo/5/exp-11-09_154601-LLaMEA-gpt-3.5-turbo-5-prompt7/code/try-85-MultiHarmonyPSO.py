import numpy as np

class MultiHarmonyPSO:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjust_rate=0.1, bandwidth=0.01, swarm_size=10, inertia_weight=0.5, cognitive_weight=1.0, social_weight=2.0):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.bandwidth = bandwidth
        self.swarm_size = swarm_size
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))

        def pitch_adjustment(harmony):
            num_adjust = int(self.pitch_adjust_rate * self.dim)
            indices = np.random.choice(self.dim, num_adjust, replace=False)
            harmony[indices] = harmony[indices] + np.random.uniform(-self.bandwidth, self.bandwidth, size=num_adjust)
            return harmony

        def update_particle(best_particle, particle, global_best):
            inertia_term = self.inertia_weight * particle['velocity']
            cognitive_term = self.cognitive_weight * np.random.rand() * (particle['best_position'] - particle['position'])
            social_term = self.social_weight * np.random.rand() * (global_best - particle['position'])
            particle['velocity'] = inertia_term + cognitive_term + social_term
            particle['position'] = particle['position'] + particle['velocity']
            return particle

        harmony_memory = initialize_harmony_memory()
        swarm = [{'position': np.random.uniform(-5.0, 5.0, size=self.dim), 'velocity': np.zeros(self.dim), 'best_position': np.zeros(self.dim)} for _ in range(self.swarm_size)]
        global_best = np.inf
        while self.budget > 0:
            new_harmony = np.mean(harmony_memory, axis=0)
            new_harmony = pitch_adjustment(new_harmony)
            if func(new_harmony) < func(harmony_memory.min(axis=0)):
                replace_idx = np.argmax(func(harmony_memory))
                harmony_memory[replace_idx] = new_harmony
                if np.random.rand() < 0.5 and self.harmony_memory_size < 20:
                    self.harmony_memory_size += 1
                    harmony_memory = np.vstack((harmony_memory, np.random.uniform(-5.0, 5.0, size=self.dim)))
                    harmony_memory = np.delete(harmony_memory, replace_idx, axis=0)
            for particle in swarm:
                particle['best_position'] = particle['position'] if func(particle['position']) < func(particle['best_position']) else particle['best_position']
                global_best = func(particle['position']) if func(particle['position']) < global_best else global_best
                particle = update_particle(global_best, particle, global_best)
            self.budget -= 1

        return harmony_memory.min(axis=0)
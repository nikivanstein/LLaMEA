class DynamicInertiaMultiHarmonyPSO(MultiHarmonyPSO):
    def __call__(self, func):
        def initialize_inertia_weights():
            return np.full(self.swarm_size, self.inertia_weight)

        inertia_weights = initialize_inertia_weights()
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
            for i, particle in enumerate(swarm):
                particle['best_position'] = particle['position'] if func(particle['position']) < func(particle['best_position']) else particle['best_position']
                global_best = func(particle['position']) if func(particle['position']) < global_best else global_best
                particle['velocity'] = inertia_weights[i] * particle['velocity']
                particle = update_particle(global_best, particle, global_best)
                inertia_weights[i] = self.inertia_weight * (1 - func(particle['position']) / global_best)
            self.budget -= 1

        return harmony_memory.min(axis=0)
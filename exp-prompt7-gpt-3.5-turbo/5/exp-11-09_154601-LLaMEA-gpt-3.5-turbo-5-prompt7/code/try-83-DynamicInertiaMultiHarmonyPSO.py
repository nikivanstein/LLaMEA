class DynamicInertiaMultiHarmonyPSO(MultiHarmonyPSO):
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjust_rate=0.1, bandwidth=0.01, swarm_size=10, inertia_weight_min=0.1, inertia_weight_max=1.0, cognitive_weight=1.0, social_weight=2.0):
        super().__init__(budget, dim, harmony_memory_size, pitch_adjust_rate, bandwidth, swarm_size, inertia_weight_min, cognitive_weight, social_weight)
        self.inertia_weight_max = inertia_weight_max

    def __call__(self, func):
        def update_particle(best_particle, particle, global_best):
            inertia_range = self.inertia_weight_max - self.inertia_weight
            inertia_weight = self.inertia_weight + np.random.rand() * inertia_range
            inertia_term = inertia_weight * particle['velocity']
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
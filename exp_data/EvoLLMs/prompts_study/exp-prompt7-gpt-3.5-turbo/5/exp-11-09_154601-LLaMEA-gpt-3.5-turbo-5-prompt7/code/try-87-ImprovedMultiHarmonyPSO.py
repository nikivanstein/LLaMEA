class ImprovedMultiHarmonyPSO(MultiHarmonyPSO):
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjust_rate=0.1, bandwidth=0.01, swarm_size=10, inertia_min=0.4, inertia_max=0.9, cognitive_weight=1.0, social_weight=2.0):
        super().__init__(budget, dim, harmony_memory_size, pitch_adjust_rate, bandwidth, swarm_size, inertia_min, cognitive_weight, social_weight)
        self.inertia_min = inertia_min
        self.inertia_max = inertia_max

    def update_particle(self, best_particle, particle, global_best):
        inertia = self.inertia_min + ((self.budget / self.max_budget) * (self.inertia_max - self.inertia_min))
        inertia_term = inertia * particle['velocity']
        cognitive_term = self.cognitive_weight * np.random.rand() * (particle['best_position'] - particle['position'])
        social_term = self.social_weight * np.random.rand() * (global_best - particle['position'])
        particle['velocity'] = inertia_term + cognitive_term + social_term
        particle['position'] = particle['position'] + particle['velocity']
        return particle
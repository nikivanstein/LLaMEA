class DynamicInertiaWeightQIPSO(AdaptiveCognitiveWeightQIPSO):
    def __init__(self, budget, dim, num_particles=30, inertia_weight_min=0.1, inertia_weight_max=0.9, inertia_weight_decay=0.95, **kwargs):
        super().__init__(budget, dim, num_particles, inertia_weight_decay=inertia_weight_decay, **kwargs)
        self.inertia_weight_min = inertia_weight_min
        self.inertia_weight_max = inertia_weight_max

    def __call__(self, func):
        inertia_weight = self.inertia_weight
        for _ in range(self.budget):
            # Update inertia weight dynamically
            inertia_weight = max(self.inertia_weight_min, inertia_weight * self.inertia_weight_decay)
            # Other optimization steps remain unchanged
            # ...
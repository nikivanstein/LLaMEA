import numpy as np

class DynamicHybridPSODE(HybridPSODE):
    def __init__(self, budget, dim, population_size=30, cognitive_weight=1.5, social_weight=1.5, scaling_factor=0.5, crossover_rate=0.9, inertia_min=0.4, inertia_max=0.9):
        super().__init__(budget, dim, population_size, inertia_weight=0.5, cognitive_weight=cognitive_weight, social_weight=social_weight, scaling_factor=scaling_factor, crossover_rate=crossover_rate)
        self.inertia_min = inertia_min
        self.inertia_max = inertia_max

    def __call__(self, func):
        self.inertia_weight = self.inertia_max
        inertia_decay = (self.inertia_max - self.inertia_min) / self.budget

        # Existing HybridPSODE code remains unchanged
        
        return g_best
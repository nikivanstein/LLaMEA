import numpy as np

class EnhancedDynamicAdaptiveHybridPSOADE(DynamicAdaptiveHybridPSOADE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.dynamic_param = np.array([0.4, 0.7, 0.9, 0.1, 0.3])  # Dynamic parameter array

    def adaptive_inertia_weight(self, fitness, global_best_fitness):
        return 0.5 + 0.3 * np.tanh(fitness - global_best_fitness) + 0.1 * np.sin(fitness)  # Enhanced inertia weight calculation

    def dynamic_chaos_mutation(self, position):
        chaos_params = np.sin(position) * np.tanh(np.linalg.norm(position))  # Enhanced dynamic chaotic mapping with fitness-based adaptation
        return chaos_params
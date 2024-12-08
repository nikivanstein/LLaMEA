import numpy as np

class EnhancedBatAlgorithmOptimizer:
    def __init__(self, budget, dim, population_size=10, loudness=0.5, pulse_rate=0.5, alpha=0.9, gamma=0.9, differential_weight=0.5, crossover_rate=0.7, crossover_adjust_rate=0.1, mutation_scale=0.1, mutation_scale_adjust_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.loudness = loudness
        self.pulse_rate = pulse_rate
        self.alpha = alpha
        self.gamma = gamma
        self.differential_weight = differential_weight
        self.crossover_rate = crossover_rate
        self.crossover_adjust_rate = crossover_adjust_rate
        self.mutation_scale = mutation_scale
        self.mutation_scale_adjust_rate = mutation_scale_adjust_rate

    def __call__(self, func):
        def update_mutation_scale(fitness_improved):
            if fitness_improved:
                return self.mutation_scale * (1.0 + self.mutation_scale_adjust_rate)
            else:
                return self.mutation_scale / (1.0 + self.mutation_scale_adjust_rate)

        # Rest of the code remains the same
        # ...
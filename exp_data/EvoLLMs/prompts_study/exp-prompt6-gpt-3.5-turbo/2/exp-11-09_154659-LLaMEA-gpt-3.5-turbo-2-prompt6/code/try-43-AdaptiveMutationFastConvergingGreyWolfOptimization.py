import numpy as np

class AdaptiveMutationFastConvergingGreyWolfOptimization(FastConvergingGreyWolfOptimization):
    def update_position(self, wolf, alpha, beta, delta, prev_fitness, fitness_landscape):
        fitness_improvement = prev_fitness - wolf['fitness']
        step_size = 1.0 - np.exp(-0.1 * fitness_improvement) * (1 + 0.2 * np.mean(fitness_landscape) / wolf['fitness')
        a = 1.8 - 1.6 * (np.arange(self.dim) / (self.dim - 1)) * step_size
        # Rest of the update_position function remains the same
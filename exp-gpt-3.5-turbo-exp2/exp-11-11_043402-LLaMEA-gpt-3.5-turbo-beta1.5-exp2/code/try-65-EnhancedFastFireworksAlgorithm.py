import numpy as np

class EnhancedFastFireworksAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        population_size = 10
        fireworks = np.random.uniform(-5.0, 5.0, size=(population_size, self.dim))
        best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
        for _ in range(self.budget // population_size - 1):
            sparks = np.random.uniform(-0.1, 0.1, size=(population_size, self.dim))
            fitness_values = [func(firework) for firework in fireworks]
            diversity_factor = np.mean(np.std(fireworks, axis=0))
            for i in range(population_size):
                for j in range(self.dim):
                    amplitude_factor = 1.0 + (fitness_values[i] - min(fitness_values)) / max(1e-6, max(fitness_values) - min(fitness_values))
                    sparks[i][j] *= diversity_factor * amplitude_factor
            fireworks += sparks
            best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
        return best_firework
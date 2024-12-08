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
            diversity_factor = np.mean(np.std(fireworks, axis=0) * np.abs(best_firework - fireworks), axis=1)  # Adaptive selection of sparks based on individual firework performance
            for i in range(population_size):
                for j in range(self.dim):
                    sparks[i][j] *= diversity_factor[i]
            fireworks += sparks
            best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
        return best_firework
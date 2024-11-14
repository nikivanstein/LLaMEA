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
            fitness_diff = np.abs(func(fireworks) - func(fireworks + sparks))
            diversity_factor = np.mean(fitness_diff)  # Dynamic mutation scaling based on fitness improvement
            for i in range(population_size):
                for j in range(self.dim):
                    sparks[i][j] *= diversity_factor
            fireworks += sparks
            best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
        return best_firework
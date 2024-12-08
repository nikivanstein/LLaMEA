import numpy as np

class EnhancedFireworksAlgorithmSpeedup:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        population_size = 10
        fireworks = np.random.uniform(-5.0, 5.0, size=(population_size, self.dim))
        best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
        for _ in range(self.budget // population_size - 1):
            sparks = np.random.uniform(-0.1, 0.1, size=(population_size, self.dim))
            scaling_factor = np.random.uniform(0.5, 1.5)  # Dynamic mutation scaling
            for i in range(population_size):
                for j in range(self.dim):
                    sparks[i][j] *= scaling_factor * np.abs(best_firework[j] - fireworks[i][j])
            fireworks += sparks
            best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
            if np.random.rand() < 0.1:  # Adaptive population size control
                population_size = np.clip(int(population_size * np.random.uniform(0.8, 1.2)), 1, 50)
                fireworks = np.vstack((fireworks, np.random.uniform(-5.0, 5.0, size=(population_size, self.dim)))
        return best_firework
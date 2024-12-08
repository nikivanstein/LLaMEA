import numpy as np

class FastFireworksAlgorithmImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        population_size = 10
        fireworks = np.random.uniform(-5.0, 5.0, size=(population_size, self.dim))
        best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
        for _ in range(self.budget // population_size - 1):
            sparks = np.random.uniform(-0.1, 0.1, size=(population_size, self.dim))
            performance = [func(firework) for firework in fireworks]
            spark_amplitudes = np.mean(np.abs(sparks), axis=1) * np.array(performance) / np.mean(performance)
            for i in range(population_size):
                for j in range(self.dim):
                    sparks[i][j] *= spark_amplitudes[i]
            fireworks += sparks
            best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
        return best_firework
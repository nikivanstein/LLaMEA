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
            pop_diversity = np.mean(np.std(fireworks, axis=0))
            ind_diversity = np.std(fireworks, axis=1)
            for i in range(population_size):
                spark_magnitude = (pop_diversity * 0.7 + ind_diversity[i] * 0.3) * np.random.uniform(0.9, 1.1)
                sparks[i] *= spark_magnitude
            fireworks += sparks
            best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
        return best_firework
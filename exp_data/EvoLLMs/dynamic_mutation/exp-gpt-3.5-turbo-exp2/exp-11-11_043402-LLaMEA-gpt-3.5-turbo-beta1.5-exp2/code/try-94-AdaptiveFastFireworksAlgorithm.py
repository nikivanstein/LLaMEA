import numpy as np

class AdaptiveFastFireworksAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        population_size = 10
        fireworks = np.random.uniform(-5.0, 5.0, size=(population_size, self.dim))
        best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
        for _ in range(self.budget // population_size - 1):
            sparks = np.random.uniform(-0.1, 0.1, size=(population_size, self.dim))
            diversity_factor = np.mean(np.std(fireworks, axis=0)
                                       * np.linspace(1.0, 0.1, num=self.budget // population_size - 1)[:, np.newaxis])
            for i in range(population_size):
                for j in range(self.dim):
                    sparks[i][j] *= diversity_factor * np.abs(best_firework[j] - fireworks[i][j])
            fireworks += sparks
            best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
            if np.random.rand() < 0.1:  # Adjust population size dynamically
                population_size = min(20, int(population_size * 1.2))
                fireworks = np.concatenate((fireworks, np.random.uniform(-5.0, 5.0, size=(population_size - fireworks.shape[0], self.dim))))
        return best_firework
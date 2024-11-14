import numpy as np

class DynamicPopulationFastFireworksAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 10
        fireworks = np.random.uniform(-5.0, 5.0, size=(population_size, self.dim))
        best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
        for _ in range(self.budget // population_size - 1):
            sparks = np.random.uniform(-0.1, 0.1, size=(population_size, self.dim))
            diversity_factor = np.mean(np.std(fireworks, axis=0) * np.random.uniform(0.9, 1.1))  # Dynamic mutation scaling based on population diversity with adaptive factor
            for i in range(population_size):
                for j in range(self.dim):
                    sparks[i][j] *= diversity_factor * np.abs(best_firework[j] - fireworks[i][j])
            fireworks += sparks
            population_size = max(2, min(20, int(np.clip(np.mean([func(firework) for firework in fireworks]) * 10, 2, 20))))  # Adaptive population size adjustment based on fitness
            fireworks = fireworks[:population_size]
            best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
        return best_firework
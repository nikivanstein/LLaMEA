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
            diversity_factor = np.mean(np.std(fireworks, axis=0) + 1e-6)  # Ensure non-zero division
            for i in range(population_size):
                sparks = np.random.uniform(-0.1, 0.1, size=self.dim) * diversity_factor * np.abs(best_firework - fireworks[i])
                new_firework = fireworks[i] + sparks
                if func(new_firework) < func(fireworks[i]):  # Update firework if spark improves fitness
                    fireworks[i] = new_firework
                    if func(new_firework) < func(best_firework):  # Update best firework if improved
                        best_firework = new_firework
        return best_firework
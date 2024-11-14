import numpy as np

class EnhancedFireworksAlgorithm:
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
                                       * np.abs(best_firework - fireworks), axis=1)  # Dynamic step size adaptation
            for i in range(population_size):
                sparks[i] *= diversity_factor[i]
            fireworks += sparks
            new_fitness = [func(firework) for firework in fireworks]
            best_firework = fireworks[np.argmin(new_fitness)]
            if new_fitness[i] < func(best_firework):
                best_firework = fireworks[i]
        return best_firework
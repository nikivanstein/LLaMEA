import numpy as np

class FastFireworksAlgorithmImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.inertia_weight = 0.5

    def __call__(self, func):
        population_size = 10
        fireworks = np.random.uniform(-5.0, 5.0, size=(population_size, self.dim))
        best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
        for _ in range(self.budget // population_size - 1):
            sparks = np.random.uniform(-0.1, 0.1, size=(population_size, self.dim))
            diversity_factor = np.mean(np.std(fireworks, axis=0) * self.inertia_weight)  
            for i in range(population_size):
                for j in range(self.dim):
                    sparks[i][j] *= diversity_factor * np.abs(best_firework[j] - fireworks[i][j])
            fireworks += sparks
            best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
            self.inertia_weight *= 0.95  # Update inertia weight for self-adaptation
        return best_firework
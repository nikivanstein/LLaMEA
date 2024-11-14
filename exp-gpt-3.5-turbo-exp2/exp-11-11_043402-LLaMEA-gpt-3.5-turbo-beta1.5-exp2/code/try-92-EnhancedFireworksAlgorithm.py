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
            fitness_scores = [func(firework) for firework in fireworks]
            sparks = np.random.uniform(-0.1, 0.1, size=(population_size, self.dim))
            diversity_factor = np.mean(np.std(fireworks, axis=0))  # Dynamic mutation scaling based on population diversity
            selected_sparks = sparks[np.argsort(fitness_scores)[:population_size // 2]]  # Select sparks based on fitness
            for i in range(population_size):
                for j in range(self.dim):
                    selected_sparks[i][j] *= diversity_factor * np.abs(best_firework[j] - fireworks[i][j])
            fireworks += selected_sparks
            best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
        return best_firework
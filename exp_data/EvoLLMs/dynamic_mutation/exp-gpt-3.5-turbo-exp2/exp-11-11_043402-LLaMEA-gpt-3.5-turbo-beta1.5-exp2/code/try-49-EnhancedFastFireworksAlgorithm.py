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
            diversity_factor = np.mean(np.std(fireworks, axis=0) + 1e-6)  # Dynamic mutation scaling based on population diversity with bias
            fitness_values = [func(firework) for firework in fireworks]
            normalized_fitness = (fitness_values - np.min(fitness_values)) / (np.max(fitness_values) - np.min(fitness_values))
            selection_probabilities = 1 - normalized_fitness
            selection_probabilities /= np.sum(selection_probabilities)
            selected_indices = np.random.choice(range(population_size), population_size, p=selection_probabilities)
            for i in range(population_size):
                for j in range(self.dim):
                    sparks[i][j] *= diversity_factor * np.abs(best_firework[j] - fireworks[selected_indices[i]][j])
            fireworks += sparks
            best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
        return best_firework
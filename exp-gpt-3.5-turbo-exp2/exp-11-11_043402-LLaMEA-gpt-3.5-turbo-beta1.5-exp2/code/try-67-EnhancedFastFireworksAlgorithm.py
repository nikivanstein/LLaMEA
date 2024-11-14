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
            
            # Dynamic selection mechanism to prioritize exploration of diverse regions
            diversity_scores = np.std(fireworks, axis=1)
            selected_indices = np.argsort(diversity_scores)[:int(population_size * 0.5)]
            selected_fireworks = fireworks[selected_indices]
            
            for i in range(population_size):
                if i not in selected_indices:
                    fireworks[i] += sparks[i]  # Continue with original sparks for non-selected fireworks
                else:
                    for j in range(self.dim):
                        sparks[i][j] *= np.abs(best_firework[j] - selected_fireworks[i][j])  # Adjust sparks based on diversity
                    fireworks[i] += sparks[i]
            
            best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
        
        return best_firework
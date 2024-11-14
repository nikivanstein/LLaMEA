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
            diversity_factor = np.mean(np.std(fireworks, axis=0))  # Dynamic mutation scaling based on population diversity
            for i in range(population_size):
                mutated_firework = fireworks[i] + 0.5 * np.random.uniform() * (fireworks[i] - best_firework)  # Differential evolution mutation
                for j in range(self.dim):
                    sparks[i][j] = diversity_factor * mutated_firework[j]  # Hybrid mutation strategy
            fireworks += sparks
            best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
        return best_firework
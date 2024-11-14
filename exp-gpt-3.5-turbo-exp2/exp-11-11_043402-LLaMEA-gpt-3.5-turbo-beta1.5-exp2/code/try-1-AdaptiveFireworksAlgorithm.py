class AdaptiveFireworksAlgorithm(FireworksAlgorithm):
    def __call__(self, func):
        population_size = 10
        fireworks = np.random.uniform(-5.0, 5.0, size=(population_size, self.dim))
        best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
        for _ in range(self.budget // population_size - 1):
            sparks = np.random.uniform(-0.1, 0.1, size=(population_size, self.dim))
            for i in range(population_size):
                mutation_rate = np.random.uniform(0.1, 0.9)  # Adaptive mutation rate
                for j in range(self.dim):
                    sparks[i][j] *= mutation_rate * np.abs(best_firework[j] - fireworks[i][j])
            fireworks += sparks
            best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
        return best_firework
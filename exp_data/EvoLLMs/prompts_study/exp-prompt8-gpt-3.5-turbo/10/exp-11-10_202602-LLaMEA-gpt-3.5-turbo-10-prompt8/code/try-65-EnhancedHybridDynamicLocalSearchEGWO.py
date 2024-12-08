import numpy as np

class EnhancedHybridDynamicLocalSearchEGWO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def update_position(position, best, a, c, mutation_rate):
            return np.clip(position + a * mutation_rate * np.random.randn(self.dim) * np.abs(c * best - position), -5.0, 5.0)

        def adaptive_mutation(x, population, f, mutation_rate):
            a, b, c = population[np.random.choice(population.shape[0], 3, replace=False)]
            return np.clip(a + f * mutation_rate * (b - c), -5.0, 5.0)

        positions = np.random.uniform(-5.0, 5.0, (5, self.dim))
        fitness = np.array([func(p) for p in positions])
        best_idx = np.argmin(fitness)
        best_position = positions[best_idx]
        mutation_rates = np.full(self.budget - 5, 1.0)  # Initialize mutation rates

        for idx in range(self.budget - 5):
            a = 2 - 2 * idx / (self.budget - 1)  # linearly decreasing a value
            mutation_rate = mutation_rates[idx]  # Use current mutation rate

            for i in range(5):
                if i == best_idx:
                    continue
                c1 = 2 * np.random.rand(self.dim)
                c2 = 2 * np.random.rand(self.dim)
                c3 = 2 * np.random.rand(self.dim)
                if np.random.rand() > 0.5:  
                    positions[i] = update_position(positions[i], best_position, c1, c2, mutation_rate)
                else:
                    positions[i] = adaptive_mutation(positions[i], positions, 0.5, mutation_rate)
                
            new_fitness = np.array([func(p) for p in positions])
            new_best_idx = np.argmin(new_fitness)
            if new_fitness[new_best_idx] < fitness[best_idx]:
                fitness[new_best_idx] = new_fitness[new_best_idx]
                best_idx = new_best_idx
                best_position = positions[best_idx]

            # Adapt mutation rate based on performance
            if np.random.rand() < 0.1:  # 10% code difference
                if new_fitness[new_best_idx] < fitness[best_idx]:
                    mutation_rates[idx] *= 1.1  # Increase mutation rate
                else:
                    mutation_rates[idx] *= 0.9  # Decrease mutation rate

        return best_position
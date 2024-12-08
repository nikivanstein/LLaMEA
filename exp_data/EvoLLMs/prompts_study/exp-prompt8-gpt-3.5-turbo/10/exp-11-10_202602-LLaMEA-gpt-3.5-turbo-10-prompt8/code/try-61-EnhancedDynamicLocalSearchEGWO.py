import numpy as np

class EnhancedDynamicLocalSearchEGWO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def update_position(position, best, a, c, adaptive_mut_prob):
            mutation_prob = np.random.rand(self.dim) < adaptive_mut_prob
            adaptive_mut = best * 0.8 + position * 0.2  # Weighted average
            return np.clip(position + a * (2 * np.random.rand(self.dim) - 1) * np.abs(c * adaptive_mut - position), -5.0, 5.0) * mutation_prob + position * (1 - mutation_prob)
        
        positions = np.random.uniform(-5.0, 5.0, (5, self.dim))
        fitness = np.array([func(p) for p in positions])
        best_idx = np.argmin(fitness)
        best_position = positions[best_idx]

        for _ in range(self.budget - 5):
            a = 2 - 2 * _ / (self.budget - 1)  # linearly decreasing a value
            adaptive_mut_prob = 0.5 + 0.5 * (self.budget - 5 - _) / (self.budget - 5)  # Adaptive mutation probability
            for i in range(5):
                if i == best_idx:
                    continue
                c1 = 2 * np.random.rand(self.dim)
                c2 = 2 * np.random.rand(self.dim)
                c3 = 2 * np.random.rand(self.dim)
                if np.random.rand() > 0.5:  
                    positions[i] = update_position(positions[i], best_position, c1, c2, adaptive_mut_prob)
                else:
                    positions[i] = update_position(positions[i], positions[best_idx], c3, c3, adaptive_mut_prob)
                    
                if np.random.rand() < 0.3:  # Introduce dynamic local search
                    positions[i] = update_position(positions[i], best_position, c1, c2, adaptive_mut_prob)
                    
            new_fitness = np.array([func(p) for p in positions])
            new_best_idx = np.argmin(new_fitness)
            if new_fitness[new_best_idx] < fitness[best_idx]:
                fitness[new_best_idx] = new_fitness[new_best_idx]
                best_idx = new_best_idx
                best_position = positions[best_idx]

        return best_position
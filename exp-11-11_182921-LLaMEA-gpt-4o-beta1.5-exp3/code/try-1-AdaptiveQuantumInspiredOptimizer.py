import numpy as np

class AdaptiveQuantumInspiredOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.probability_amplitudes = np.random.uniform(0, 1, (self.population_size, dim))
        self.best_global_position = np.copy(self.positions[0])
        self.func_evaluations = 0
        self.best_score = float('inf')

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            scores = np.array([func(pos) for pos in self.positions])
            self.func_evaluations += self.population_size

            min_index = np.argmin(scores)
            if scores[min_index] < self.best_score:
                self.best_global_position = self.positions[min_index]
                self.best_score = scores[min_index]

            exploration_intensity = 0.5 * (1 - self.func_evaluations / self.budget)
            
            for i in range(self.population_size):
                quantum_state = np.sign(np.random.uniform(-1, 1, self.dim)) * self.probability_amplitudes[i]
                self.positions[i] += exploration_intensity * quantum_state * (self.best_global_position - self.positions[i])
                
                # Ensure positions remain within bounds
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

                # Update probability amplitudes for adaptive exploration
                self.probability_amplitudes[i] = np.random.uniform(0, 1, self.dim) * exploration_intensity

        return self.best_global_position
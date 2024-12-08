import numpy as np

class QuantumInspiredAdaptiveSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.temperature = 1.0
        self.func_evaluations = 0
        self.alpha = 0.98  # Cooling rate
        self.position = np.random.uniform(self.lower_bound, self.upper_bound, dim)
        self.best_position = np.copy(self.position)
        self.best_score = float('inf')

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Quantum-inspired superposition: generate new candidate solutions
            superposed_state = np.random.uniform(-1, 1, self.dim) * self.temperature
            candidate_position = self.position + superposed_state
            candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)
            
            # Evaluate candidate solution
            candidate_score = func(candidate_position)
            self.func_evaluations += 1
            
            # Accept or reject the new candidate
            if candidate_score < self.best_score or np.random.rand() < np.exp((self.best_score - candidate_score) / self.temperature):
                self.position = candidate_position
                if candidate_score < self.best_score:
                    self.best_score = candidate_score
                    self.best_position = candidate_position

            # Adaptive cooling schedule
            self.temperature = max(0.01, self.alpha * (1 - self.func_evaluations / self.budget) * self.temperature)

        return self.best_position
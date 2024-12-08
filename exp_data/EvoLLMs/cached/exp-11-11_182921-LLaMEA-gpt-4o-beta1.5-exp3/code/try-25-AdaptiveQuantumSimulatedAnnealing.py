import numpy as np

class AdaptiveQuantumSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.func_evaluations = 0
        self.temperature = 1.0  # Initial temperature
        self.cooling_rate = 0.99  # Cooling rate
        self.tau = 0.1  # Quantum tunneling probability
        self.current_position = np.random.uniform(self.lower_bound, self.upper_bound, dim)
        self.best_position = np.copy(self.current_position)
        self.best_score = float('inf')

    def __call__(self, func):
        current_score = func(self.current_position)
        self.func_evaluations += 1
        if current_score < self.best_score:
            self.best_score = current_score
            self.best_position = np.copy(self.current_position)

        while self.func_evaluations < self.budget:
            # Generate a neighbor solution
            neighbor = self.current_position + np.random.normal(0, self.temperature, self.dim)
            neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)

            # Quantum tunneling
            if np.random.rand() < self.tau:
                quantum_jump = np.random.normal(0, 1, self.dim)
                neighbor += quantum_jump
                neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)

            neighbor_score = func(neighbor)
            self.func_evaluations += 1

            # Acceptance criteria
            if neighbor_score < current_score or np.exp((current_score - neighbor_score) / self.temperature) > np.random.rand():
                self.current_position = neighbor
                current_score = neighbor_score

                if current_score < self.best_score:
                    self.best_score = current_score
                    self.best_position = np.copy(self.current_position)

            # Adaptive cooling schedule
            self.temperature *= self.cooling_rate ** (self.func_evaluations / self.budget)

        return self.best_position
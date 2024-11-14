import numpy as np

class QuantumInspiredEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.q_population = np.pi * np.random.uniform(-1, 1, (self.population_size, dim))  # Quantum states (angles)
        self.best_solution = None
        self.best_score = float('inf')
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Convert quantum states to real values via quantum observation (superposition)
            real_population = self.lower_bound + (self.upper_bound - self.lower_bound) * (0.5 * (1 + np.cos(self.q_population)))

            # Evaluate
            scores = np.array([func(ind) for ind in real_population])
            self.func_evaluations += self.population_size

            # Update best solution
            min_index = np.argmin(scores)
            if scores[min_index] < self.best_score:
                self.best_solution = real_population[min_index]
                self.best_score = scores[min_index]

            # Quantum interference: update using the best found solution
            best_angles = np.arccos((2 * (self.best_solution - self.lower_bound) / (self.upper_bound - self.lower_bound)) - 1)
            self.q_population += np.random.uniform(-0.1, 0.1, (self.population_size, self.dim)) * (best_angles - self.q_population)

            # Maintain boundaries in quantum states
            self.q_population = np.clip(self.q_population, -np.pi, np.pi)

        return self.best_solution
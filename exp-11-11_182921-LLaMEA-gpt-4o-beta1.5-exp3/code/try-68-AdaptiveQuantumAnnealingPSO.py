import numpy as np

class AdaptiveQuantumAnnealingPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.func_evaluations = 0
        self.best_score = float('inf')
        self.best_position = None
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_annealing_factor = 0.9
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.quantum_probability = 0.05

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            for i in range(self.population_size):
                # Evaluate current particle
                current_score = func(self.population[i])
                self.func_evaluations += 1

                # Update personal best
                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = self.population[i]

                # Update global best
                if current_score < self.best_score:
                    self.best_score = current_score
                    self.best_position = self.population[i]

            # Update velocities and positions
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                self.velocities[i] = (
                    self.global_annealing_factor * self.velocities[i]
                    + self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
                    + self.c2 * r2 * (self.best_position - self.population[i])
                )
                
                # Quantum-inspired perturbation
                if np.random.rand() < self.quantum_probability:
                    self.velocities[i] += np.random.normal(0, 1, self.dim)

                # Update position
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

            # Annealing factor adaptation
            self.global_annealing_factor *= 0.99

        return self.best_position
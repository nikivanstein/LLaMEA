import numpy as np

class QiPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.pop_size, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.evaluations = 0
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5 # Cognitive coefficient
        self.c2 = 1.5 # Social coefficient

    def evaluate(self, func, position):
        self.evaluations += 1
        return func(position)

    def quantum_superposition(self, pos):
        return pos + np.random.uniform(-1, 1, pos.shape) * np.sin(np.random.uniform(-np.pi, np.pi, pos.shape))

    def __call__(self, func):
        np.random.seed(42)

        # Initial evaluation
        for i in range(self.pop_size):
            score = self.evaluate(func, self.positions[i])
            self.personal_best_scores[i] = score
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.positions[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.positions[i]) +
                                      self.c2 * np.random.rand(self.dim) * (self.global_best_position - self.positions[i]))

                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

                # Quantum superposition step
                quantum_position = self.quantum_superposition(self.positions[i])
                quantum_position = np.clip(quantum_position, self.lower_bound, self.upper_bound)

                # Fitness evaluation
                score = self.evaluate(func, quantum_position)
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = quantum_position
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = quantum_position

        return self.global_best_position
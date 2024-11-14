import numpy as np

class QIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.position = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_position = np.copy(self.position)
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.w = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.beta = 0.5   # Quantum factor

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def quantum_position_update(self, position, global_best):
        q_position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        return self.beta * q_position + (1 - self.beta) * global_best

    def __call__(self, func):
        np.random.seed(42)

        # Initial evaluation
        for i in range(self.pop_size):
            fitness = self.evaluate(func, self.position[i])
            self.personal_best_fitness[i] = fitness
            self.personal_best_position[i] = self.position[i]
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.position[i]

        # Main optimization loop
        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_component = self.c1 * r1 * (self.personal_best_position[i] - self.position[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.position[i])
                self.velocity[i] = self.w * self.velocity[i] + cognitive_component + social_component

                if np.random.rand() < 0.5:  # Quantum position update
                    self.position[i] = self.quantum_position_update(self.position[i], self.global_best_position)
                else:
                    self.position[i] += self.velocity[i]

                self.position[i] = np.clip(self.position[i], self.lower_bound, self.upper_bound)

                # Evaluate the new position
                fitness = self.evaluate(func, self.position[i])
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_position[i] = self.position[i]
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = self.position[i]

        return self.global_best_position
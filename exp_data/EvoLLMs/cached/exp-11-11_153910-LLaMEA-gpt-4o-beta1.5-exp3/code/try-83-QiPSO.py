import numpy as np

class QiPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, float('inf'))

        self.personal_best_positions = np.copy(self.population)
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))

        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.evaluations = 0

        self.alpha = 0.75  # Quantum-inspired factor
        self.beta = 0.25   # Cognitive and social factor
        self.w = 0.5       # Inertia weight

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def __call__(self, func):
        np.random.seed(42)

        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            if self.fitness[i] < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = self.fitness[i]
                self.personal_best_positions[i] = self.population[i]
            if self.fitness[i] < self.global_best_fitness:
                self.global_best_fitness = self.fitness[i]
                self.global_best_position = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                # Update velocity with quantum-inspired term
                quantum_term = self.alpha * (np.random.rand(self.dim) - 0.5) * (self.global_best_position - self.population[i])
                cognitive_term = self.beta * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.population[i])
                social_term = self.beta * np.random.rand(self.dim) * (self.global_best_position - self.population[i])
                
                self.velocities[i] = self.w * self.velocities[i] + cognitive_term + social_term + quantum_term
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

                current_fitness = self.evaluate(func, self.population[i])
                if current_fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = current_fitness
                    self.personal_best_positions[i] = self.population[i]
                if current_fitness < self.global_best_fitness:
                    self.global_best_fitness = current_fitness
                    self.global_best_position = self.population[i]

        return self.global_best_position
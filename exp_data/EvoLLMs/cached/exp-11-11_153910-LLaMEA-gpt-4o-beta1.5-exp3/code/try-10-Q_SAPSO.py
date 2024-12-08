import numpy as np

class Q_SAPSO:
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
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.5  # Social component

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def quantum_behavior(self, position):
        """Simulate quantum behavior by randomly generating a new solution around the best global position."""
        quantum_position = np.copy(position)
        mask = np.random.rand(self.dim) < 0.5
        quantum_position[mask] = self.best_global_position[mask] + \
                                 np.random.uniform(-1, 1, np.sum(mask)) * np.abs(self.best_global_position[mask] - position[mask])
        return np.clip(quantum_position, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            if self.fitness[i] < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = self.fitness[i]
                self.personal_best_positions[i] = self.population[i]
            if self.fitness[i] < self.best_global_fitness:
                self.best_global_fitness = self.fitness[i]
                self.best_global_position = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
                social = self.c2 * r2 * (self.best_global_position - self.population[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social

                # Update position
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

                # Quantum behavior to escape local optima
                if np.random.rand() < 0.1:
                    self.population[i] = self.quantum_behavior(self.population[i])
                
                # Evaluate new position
                self.fitness[i] = self.evaluate(func, self.population[i])
                if self.fitness[i] < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = self.fitness[i]
                    self.personal_best_positions[i] = self.population[i]
                if self.fitness[i] < self.best_global_fitness:
                    self.best_global_fitness = self.fitness[i]
                    self.best_global_position = self.population[i]

        return self.best_global_position
import numpy as np

class QIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.evaluations = 0
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w = 0.7   # Inertia weight

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def quantum_particle_update(self, position, velocity):
        quantum_position = np.empty_like(position)
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                quantum_position[i] = position[i] + np.random.normal(0, np.abs(velocity[i]))
            else:
                quantum_position[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return np.clip(quantum_position, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            fitness = self.evaluate(func, self.population[i])
            self.personal_best_fitness[i] = fitness
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                # Update velocities
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.personal_best_positions[i] - self.population[i]) +
                                      self.c2 * r2 * (self.global_best_position - self.population[i]))
                # Update positions
                quantum_position = self.quantum_particle_update(self.population[i], self.velocities[i])
                fitness = self.evaluate(func, quantum_position)

                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = quantum_position
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = quantum_position
                        
            self.population = self.personal_best_positions.copy()

        return self.global_best_position
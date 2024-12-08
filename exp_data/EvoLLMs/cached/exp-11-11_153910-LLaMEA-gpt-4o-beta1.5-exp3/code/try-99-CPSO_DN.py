import numpy as np

class CPSO_DN:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.swarm_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_fitness = np.full(self.swarm_size, float('inf'))
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5 # Cognitive coefficient
        self.c2 = 1.5 # Social coefficient

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def dynamic_neighbors(self, index):
        neighborhood_size = np.random.randint(2, self.swarm_size)
        neighbors = np.random.choice(self.swarm_size, neighborhood_size, replace=False)
        return neighbors if index not in neighbors else self.dynamic_neighbors(index)

    def __call__(self, func):
        np.random.seed(42)

        # Initial evaluation
        for i in range(self.swarm_size):
            fitness = self.evaluate(func, self.positions[i])
            self.personal_best_fitness[i] = fitness
            self.personal_best_positions[i] = self.positions[i]
            if fitness < self.best_global_fitness:
                self.best_global_fitness = fitness
                self.best_global_position = self.positions[i]

        while self.evaluations < self.budget:
            for i in range(self.swarm_size):
                neighbors = self.dynamic_neighbors(i)
                best_neighbor_fitness = float('inf')
                best_neighbor_position = None

                for neighbor in neighbors:
                    if self.personal_best_fitness[neighbor] < best_neighbor_fitness:
                        best_neighbor_fitness = self.personal_best_fitness[neighbor]
                        best_neighbor_position = self.personal_best_positions[neighbor]

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) +
                                      self.c2 * r2 * (best_neighbor_position - self.positions[i]))

                self.positions[i] = self.positions[i] + self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

                fitness = self.evaluate(func, self.positions[i])
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]
                    if fitness < self.best_global_fitness:
                        self.best_global_fitness = fitness
                        self.best_global_position = self.positions[i]

        return self.best_global_position
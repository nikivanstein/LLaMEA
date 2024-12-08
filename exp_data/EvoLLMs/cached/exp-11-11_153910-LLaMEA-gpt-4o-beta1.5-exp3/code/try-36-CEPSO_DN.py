import numpy as np

class CEPSO_DN:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.best_local_positions = self.population.copy()
        self.best_local_fitness = np.full(self.pop_size, float('inf'))
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5 # Cognitive coefficient
        self.c2 = 1.5 # Social coefficient

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def dynamic_neighborhood(self, i):
        radius = max(1, int(self.pop_size * 0.1))
        distances = np.linalg.norm(self.population - self.population[i], axis=1)
        neighbors = np.argsort(distances)[:radius]
        return neighbors

    def __call__(self, func):
        np.random.seed(42)

        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            self.best_local_positions[i] = self.population[i]
            self.best_local_fitness[i] = self.fitness[i]
            if self.fitness[i] < self.best_global_fitness:
                self.best_global_fitness = self.fitness[i]
                self.best_global_position = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                neighbors = self.dynamic_neighborhood(i)

                local_best_neighbor = neighbors[np.argmin(self.best_local_fitness[neighbors])]
                local_best_position = self.best_local_positions[local_best_neighbor]
                
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + self.c1 * r1 * (self.best_local_positions[i] - self.population[i])
                    + self.c2 * r2 * (local_best_position - self.population[i])
                )
                self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lower_bound, self.upper_bound)
                
                current_fitness = self.evaluate(func, self.population[i])
                if current_fitness < self.best_local_fitness[i]:
                    self.best_local_fitness[i] = current_fitness
                    self.best_local_positions[i] = self.population[i]
                    if current_fitness < self.best_global_fitness:
                        self.best_global_fitness = current_fitness
                        self.best_global_position = self.population[i]

        return self.best_global_position
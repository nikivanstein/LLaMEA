import numpy as np

class MSPSO_AIW:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 3
        self.swarm_size = 10
        self.pop_size = self.num_swarms * self.swarm_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.best_personal_positions = np.copy(self.population)
        self.best_personal_fitness = np.full(self.pop_size, float('inf'))
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0
        self.w = 0.9  # Initial inertia weight
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def update_inertia(self):
        self.w = 0.4 + 0.5 * (self.budget - self.evaluations) / self.budget

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            if self.fitness[i] < self.best_personal_fitness[i]:
                self.best_personal_fitness[i] = self.fitness[i]
                self.best_personal_positions[i] = self.population[i]
            if self.fitness[i] < self.best_global_fitness:
                self.best_global_fitness = self.fitness[i]
                self.best_global_position = self.population[i]

        while self.evaluations < self.budget:
            self.update_inertia()
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(2, self.dim)
                cognitive = self.c1 * r1 * (self.best_personal_positions[i] - self.population[i])
                social = self.c2 * r2 * (self.best_global_position - self.population[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

                fitness = self.evaluate(func, self.population[i])
                if fitness < self.best_personal_fitness[i]:
                    self.best_personal_fitness[i] = fitness
                    self.best_personal_positions[i] = self.population[i]
                if fitness < self.best_global_fitness:
                    self.best_global_fitness = fitness
                    self.best_global_position = self.population[i]

        return self.best_global_position
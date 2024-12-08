import numpy as np

class AQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.evaluations = 0
        self.w = 0.9  # Initial inertia weight
        self.c1 = 1.49445  # cognitive component
        self.c2 = 1.49445  # social component

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def update_inertia_weight(self):
        self.w = 0.4 + 0.5 * (self.budget - self.evaluations) / self.budget

    def quantum_position_update(self, position, personal_best, global_best):
        beta = np.random.uniform(0, 1, self.dim)
        position = beta * personal_best + (1 - beta) * global_best
        return position

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
            self.update_inertia_weight()
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.population[i])
                self.velocities[i] = (self.w * self.velocities[i] + cognitive_component + social_component)
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

                fitness = self.evaluate(func, self.population[i])
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = self.population[i]
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = self.population[i]

                if np.random.rand() < 0.1:  # Quantum-inspired update with a small probability
                    self.population[i] = self.quantum_position_update(self.population[i], self.personal_best_positions[i], self.global_best_position)
                    self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

        return self.global_best_position
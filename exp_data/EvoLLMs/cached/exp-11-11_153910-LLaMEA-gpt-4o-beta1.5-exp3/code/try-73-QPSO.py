import numpy as np

class QPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.pop_size, self.dim))
        self.personal_best_positions = self.population.copy()
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.evaluations = 0
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5 # Cognitive coefficient
        self.c2 = 1.5 # Social coefficient

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def update_velocities_and_positions(self):
        for i in range(self.pop_size):
            r1, r2 = np.random.rand(), np.random.rand()
            cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
            social_component = self.c2 * r2 * (self.global_best_position - self.population[i])
            self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
            self.population[i] = self.population[i] + self.velocities[i]
            self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

    def quantum_update(self):
        beta = np.random.uniform(0, 1, self.dim)
        mbest = np.mean(self.population, axis=0)
        for i in range(self.pop_size):
            u = np.random.uniform(0, 1, self.dim)
            if np.random.rand() > 0.5:
                self.population[i] = mbest + beta * np.abs(self.global_best_position - self.population[i]) * np.log(1 / u)
            else:
                self.population[i] = mbest - beta * np.abs(self.global_best_position - self.population[i]) * np.log(1 / u)
            self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            fitness = self.evaluate(func, self.population[i])
            self.personal_best_fitness[i] = fitness
            self.personal_best_positions[i] = self.population[i]
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.population[i]

        while self.evaluations < self.budget:
            self.update_velocities_and_positions()
            self.quantum_update()

            for i in range(self.pop_size):
                fitness = self.evaluate(func, self.population[i])
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = self.population[i]
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = self.population[i]

        return self.global_best_position